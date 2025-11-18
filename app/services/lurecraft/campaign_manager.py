"""
Campaign Manager
Manage phishing campaigns from planning to execution
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
import structlog

from .payload_generator import PayloadGenerator

logger = structlog.get_logger()


class Campaign:
    """
    Represents a phishing campaign
    """

    def __init__(
        self,
        campaign_name: str,
        template: str,
        payload_url: str,
        targets: List[Dict[str, str]]
    ):
        self.campaign_id = str(uuid.uuid4())
        self.campaign_name = campaign_name
        self.template = template
        self.payload_url = payload_url
        self.targets = targets

        self.created_at = datetime.now(timezone.utc)
        self.status = "pending"  # pending, active, paused, completed

        self.emails_sent = 0
        self.payloads_delivered = 0
        self.clicks = 0
        self.compromises = 0

        self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert campaign to dictionary"""
        return {
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "template": self.template,
            "payload_url": self.payload_url,
            "target_count": len(self.targets),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "emails_sent": self.emails_sent,
            "payloads_delivered": self.payloads_delivered,
            "clicks": self.clicks,
            "compromises": self.compromises,
            "success_rate": self.calculate_success_rate(),
            "metadata": self.metadata
        }

    def calculate_success_rate(self) -> float:
        """Calculate campaign success rate"""
        if self.emails_sent == 0:
            return 0.0

        return (self.compromises / self.emails_sent) * 100


class CampaignManager:
    """
    Manage multiple phishing campaigns
    """

    def __init__(self):
        self.campaigns: Dict[str, Campaign] = {}
        self.payload_generator = PayloadGenerator()

    def create_campaign(
        self,
        campaign_name: str,
        template: str,
        payload_url: str,
        targets: List[Dict[str, str]]
    ) -> Campaign:
        """
        Create a new phishing campaign

        Args:
            campaign_name: Campaign name
            template: Payload template to use
            payload_url: URL to payload
            targets: List of target dicts (name, email, etc.)

        Returns:
            Campaign object
        """
        logger.info(
            "Creating campaign",
            name=campaign_name,
            template=template,
            targets=len(targets)
        )

        campaign = Campaign(
            campaign_name=campaign_name,
            template=template,
            payload_url=payload_url,
            targets=targets
        )

        self.campaigns[campaign.campaign_id] = campaign

        logger.info("Campaign created", campaign_id=campaign.campaign_id)

        return campaign

    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign by ID"""
        return self.campaigns.get(campaign_id)

    def get_all_campaigns(self) -> List[Campaign]:
        """Get all campaigns"""
        return list(self.campaigns.values())

    def update_campaign_status(self, campaign_id: str, status: str):
        """Update campaign status"""
        campaign = self.get_campaign(campaign_id)

        if campaign:
            campaign.status = status
            logger.info("Campaign status updated", campaign_id=campaign_id, status=status)

    def record_email_sent(self, campaign_id: str):
        """Record email sent"""
        campaign = self.get_campaign(campaign_id)

        if campaign:
            campaign.emails_sent += 1

    def record_payload_delivered(self, campaign_id: str):
        """Record payload delivered"""
        campaign = self.get_campaign(campaign_id)

        if campaign:
            campaign.payloads_delivered += 1

    def record_click(self, campaign_id: str):
        """Record click on payload link"""
        campaign = self.get_campaign(campaign_id)

        if campaign:
            campaign.clicks += 1
            logger.info("Click recorded", campaign_id=campaign_id, total=campaign.clicks)

    def record_compromise(self, campaign_id: str):
        """Record successful compromise"""
        campaign = self.get_campaign(campaign_id)

        if campaign:
            campaign.compromises += 1
            logger.info(
                "Compromise recorded",
                campaign_id=campaign_id,
                total=campaign.compromises,
                success_rate=campaign.calculate_success_rate()
            )

    def generate_payloads(
        self,
        campaign_id: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate payloads for campaign

        Args:
            campaign_id: Campaign ID
            output_dir: Output directory

        Returns:
            Payload manifest
        """
        campaign = self.get_campaign(campaign_id)

        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        logger.info("Generating payloads for campaign", campaign_id=campaign_id)

        manifest = self.payload_generator.generate_payload_package(
            campaign_name=campaign.campaign_name,
            targets=campaign.targets,
            payload_url=campaign.payload_url,
            template=campaign.template,
            output_dir=output_dir
        )

        return manifest

    def get_campaign_stats(self) -> Dict[str, Any]:
        """Get overall campaign statistics"""
        all_campaigns = self.get_all_campaigns()

        total_emails = sum(c.emails_sent for c in all_campaigns)
        total_clicks = sum(c.clicks for c in all_campaigns)
        total_compromises = sum(c.compromises for c in all_campaigns)

        return {
            "total_campaigns": len(all_campaigns),
            "active_campaigns": len([c for c in all_campaigns if c.status == "active"]),
            "total_emails_sent": total_emails,
            "total_clicks": total_clicks,
            "total_compromises": total_compromises,
            "overall_success_rate": (total_compromises / total_emails * 100) if total_emails > 0 else 0.0,
            "campaigns": [c.to_dict() for c in all_campaigns]
        }

    def load_targets_from_csv(self, csv_file: str) -> List[Dict[str, str]]:
        """
        Load targets from CSV file

        CSV format: name,email,company,title

        Args:
            csv_file: Path to CSV file

        Returns:
            List of target dicts
        """
        import csv

        targets = []

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    target = {
                        "name": row.get("name", "User"),
                        "email": row.get("email", "unknown@example.com"),
                        "company": row.get("company", ""),
                        "title": row.get("title", "")
                    }
                    targets.append(target)

            logger.info("Targets loaded from CSV", file=csv_file, count=len(targets))

        except Exception as e:
            logger.error("Error loading CSV", file=csv_file, error=str(e))

        return targets

    def generate_target_template(self, output_file: str = "targets_template.csv"):
        """
        Generate CSV template for targets

        Args:
            output_file: Output file path
        """
        template = """name,email,company,title
John Doe,john.doe@example.com,ACME Corp,Senior Engineer
Jane Smith,jane.smith@example.com,ACME Corp,Project Manager
Bob Johnson,bob.johnson@example.com,ACME Corp,IT Administrator
"""

        with open(output_file, 'w') as f:
            f.write(template)

        logger.info("Target template generated", file=output_file)

        return output_file
