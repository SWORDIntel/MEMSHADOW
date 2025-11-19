"""
Dynamic Web Crawler Agent

Selenium-based crawler for JavaScript-rendered content and interactive web applications.
Inspired by VantaBlackWidow's Selenium parser.

Capabilities:
- Dynamic content discovery
- JavaScript execution
- Form interaction
- AJAX endpoint detection
- Screenshot capture
- DOM analysis
"""

import uuid
import json
import base64
from typing import Dict, Any, List, Set
from base_agent import BaseAgent
import structlog

logger = structlog.get_logger()


# Selenium imports (will be installed in container)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available - crawler will run in limited mode")


class AgentCrawler(BaseAgent):
    """
    Dynamic web crawler agent using Selenium for JavaScript-rendered content
    """

    def __init__(self, agent_id: str = None):
        agent_id = agent_id or f"crawler-{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id=agent_id, agent_type="crawler")

        self.driver = None
        self.visited_urls: Set[str] = set()

        logger.info("Dynamic crawler agent initialized", agent_id=self.agent_id, selenium_available=SELENIUM_AVAILABLE)

    def _init_driver(self, headless: bool = True):
        """
        Initialize Selenium WebDriver

        Args:
            headless: Run in headless mode
        """
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not available")

        options = Options()

        if headless:
            options.add_argument('--headless')

        # Security and performance options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-notifications')

        # User agent
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)

    async def execute_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute dynamic crawling task

        Task Payload:
            - start_url: URL to start crawling
            - max_depth: Maximum crawl depth (default: 2)
            - max_pages: Maximum pages to crawl (default: 50)
            - capture_screenshots: Capture screenshots (default: False)
            - extract_forms: Extract and analyze forms (default: True)
            - detect_ajax: Detect AJAX endpoints (default: True)

        Returns:
            Crawl results with discovered URLs, forms, and endpoints
        """
        start_url = task_payload.get('start_url')
        max_depth = task_payload.get('max_depth', 2)
        max_pages = task_payload.get('max_pages', 50)
        capture_screenshots = task_payload.get('capture_screenshots', False)
        extract_forms = task_payload.get('extract_forms', True)
        detect_ajax = task_payload.get('detect_ajax', True)

        if not start_url:
            raise ValueError("start_url is required")

        logger.info(
            "Starting dynamic crawl",
            agent_id=self.agent_id,
            start_url=start_url,
            max_depth=max_depth
        )

        try:
            # Initialize driver
            self._init_driver(headless=True)

            # Crawl results
            discovered_urls = []
            forms_found = []
            ajax_endpoints = []
            screenshots = []

            # BFS crawl
            to_visit = [(start_url, 0)]  # (url, depth)
            pages_crawled = 0

            while to_visit and pages_crawled < max_pages:
                url, depth = to_visit.pop(0)

                if url in self.visited_urls or depth > max_depth:
                    continue

                # Visit page
                page_result = await self._crawl_page(
                    url,
                    extract_forms=extract_forms,
                    detect_ajax=detect_ajax,
                    capture_screenshot=capture_screenshots
                )

                self.visited_urls.add(url)
                pages_crawled += 1

                # Store results
                discovered_urls.append({
                    'url': url,
                    'depth': depth,
                    'title': page_result.get('title'),
                    'status': page_result.get('status')
                })

                if page_result.get('forms'):
                    forms_found.extend(page_result['forms'])

                if page_result.get('ajax_endpoints'):
                    ajax_endpoints.extend(page_result['ajax_endpoints'])

                if page_result.get('screenshot'):
                    screenshots.append({
                        'url': url,
                        'screenshot': page_result['screenshot']
                    })

                # Add new links to queue
                for link in page_result.get('links', []):
                    if link not in self.visited_urls:
                        to_visit.append((link, depth + 1))

            logger.info(
                "Dynamic crawl completed",
                agent_id=self.agent_id,
                pages_crawled=pages_crawled,
                forms_found=len(forms_found),
                ajax_endpoints=len(ajax_endpoints)
            )

            return {
                "discovered_urls": discovered_urls,
                "num_pages_crawled": pages_crawled,
                "forms_found": forms_found,
                "ajax_endpoints": ajax_endpoints,
                "screenshots": screenshots if capture_screenshots else [],
                "crawl_summary": f"Crawled {pages_crawled} pages, found {len(forms_found)} forms and {len(ajax_endpoints)} AJAX endpoints"
            }

        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

    async def _crawl_page(
        self,
        url: str,
        extract_forms: bool = True,
        detect_ajax: bool = True,
        capture_screenshot: bool = False
    ) -> Dict[str, Any]:
        """
        Crawl a single page with Selenium

        Args:
            url: URL to crawl
            extract_forms: Extract forms
            detect_ajax: Detect AJAX endpoints
            capture_screenshot: Capture screenshot

        Returns:
            Page crawl results
        """
        result = {
            'url': url,
            'status': 'success',
            'title': '',
            'links': [],
            'forms': [],
            'ajax_endpoints': [],
            'screenshot': None
        }

        try:
            # Navigate to page
            self.driver.get(url)

            # Wait for page load
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )

            # Get title
            result['title'] = self.driver.title

            # Extract links
            link_elements = self.driver.find_elements(By.TAG_NAME, 'a')
            for link in link_elements:
                try:
                    href = link.get_attribute('href')
                    if href and href.startswith('http'):
                        result['links'].append(href)
                except:
                    pass

            # Extract forms
            if extract_forms:
                result['forms'] = self._extract_forms()

            # Detect AJAX endpoints
            if detect_ajax:
                result['ajax_endpoints'] = self._detect_ajax_endpoints()

            # Capture screenshot
            if capture_screenshot:
                screenshot_png = self.driver.get_screenshot_as_png()
                result['screenshot'] = base64.b64encode(screenshot_png).decode()

        except TimeoutException:
            result['status'] = 'timeout'
            logger.warning(f"Page load timeout: {url}")

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Failed to crawl page {url}: {str(e)}")

        return result

    def _extract_forms(self) -> List[Dict[str, Any]]:
        """
        Extract all forms from the current page

        Returns:
            List of form dictionaries
        """
        forms = []

        try:
            form_elements = self.driver.find_elements(By.TAG_NAME, 'form')

            for idx, form in enumerate(form_elements):
                form_data = {
                    'index': idx,
                    'action': form.get_attribute('action'),
                    'method': form.get_attribute('method') or 'GET',
                    'inputs': []
                }

                # Extract input fields
                inputs = form.find_elements(By.TAG_NAME, 'input')
                for inp in inputs:
                    input_data = {
                        'name': inp.get_attribute('name'),
                        'type': inp.get_attribute('type') or 'text',
                        'value': inp.get_attribute('value'),
                        'required': inp.get_attribute('required') is not None
                    }
                    form_data['inputs'].append(input_data)

                # Extract textareas
                textareas = form.find_elements(By.TAG_NAME, 'textarea')
                for ta in textareas:
                    form_data['inputs'].append({
                        'name': ta.get_attribute('name'),
                        'type': 'textarea',
                        'required': ta.get_attribute('required') is not None
                    })

                # Extract selects
                selects = form.find_elements(By.TAG_NAME, 'select')
                for sel in selects:
                    options = [opt.get_attribute('value') for opt in sel.find_elements(By.TAG_NAME, 'option')]
                    form_data['inputs'].append({
                        'name': sel.get_attribute('name'),
                        'type': 'select',
                        'options': options,
                        'required': sel.get_attribute('required') is not None
                    })

                forms.append(form_data)

        except Exception as e:
            logger.error(f"Failed to extract forms: {str(e)}")

        return forms

    def _detect_ajax_endpoints(self) -> List[Dict[str, Any]]:
        """
        Detect AJAX endpoints by monitoring network requests

        Returns:
            List of detected AJAX endpoints
        """
        ajax_endpoints = []

        try:
            # Enable Chrome DevTools Protocol for network monitoring
            # This requires additional setup and is a simplified version

            # Alternative: Parse JavaScript for API calls
            scripts = self.driver.find_elements(By.TAG_NAME, 'script')

            for script in scripts:
                script_content = script.get_attribute('innerHTML')
                if not script_content:
                    continue

                # Look for common AJAX patterns
                import re

                # XMLHttpRequest patterns
                xhr_matches = re.findall(r'\.open\(["\']([A-Z]+)["\']\s*,\s*["\']([^"\']+)["\']', script_content)
                for method, url in xhr_matches:
                    ajax_endpoints.append({
                        'method': method,
                        'url': url,
                        'type': 'XMLHttpRequest'
                    })

                # Fetch API patterns
                fetch_matches = re.findall(r'fetch\(["\']([^"\']+)["\']', script_content)
                for url in fetch_matches:
                    ajax_endpoints.append({
                        'method': 'GET',  # Default, may be POST
                        'url': url,
                        'type': 'fetch'
                    })

                # Axios patterns
                axios_matches = re.findall(r'axios\.([a-z]+)\(["\']([^"\']+)["\']', script_content)
                for method, url in axios_matches:
                    ajax_endpoints.append({
                        'method': method.upper(),
                        'url': url,
                        'type': 'axios'
                    })

        except Exception as e:
            logger.error(f"Failed to detect AJAX endpoints: {str(e)}")

        # Deduplicate
        seen = set()
        unique_endpoints = []
        for endpoint in ajax_endpoints:
            key = (endpoint['method'], endpoint['url'])
            if key not in seen:
                seen.add(key)
                unique_endpoints.append(endpoint)

        return unique_endpoints


if __name__ == "__main__":
    agent = AgentCrawler()
    agent.run()
