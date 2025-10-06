from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from typing import Set, Callable

# This is a proof-of-concept implementation based on docs/PHASE2.MD.
# The logic for identifying sensitive paths and checking recent MFA
# status would be more sophisticated in a production system.

# Define a set of paths that are considered sensitive and require step-up auth.
SENSITIVE_PATHS: Set[str] = {
    "/api/v1/admin/users",
    "/api/v1/admin/system-report",
    "/api/v1/users/me/export-data",
}

class StepUpAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Intercepts requests to check if they require step-up authentication.
        """
        if self._requires_step_up(request):
            # In a real implementation, we would inspect a custom claim in the JWT
            # that indicates the timestamp of the last MFA authentication.
            token = request.headers.get("Authorization")

            if not self._has_recent_mfa(token):
                # If the path is sensitive and there's no recent MFA,
                # return a 401 Unauthorized with a specific error message.
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "step_up_required",
                        "detail": {
                            "reason": "This action requires recent multi-factor authentication.",
                            "challenge_types": ["fido2"], # Indicate which MFA methods are supported
                        }
                    }
                )

        # If no step-up is required, proceed with the request.
        response = await call_next(request)
        return response

    def _requires_step_up(self, request: Request) -> bool:
        """
        Determines if the request path is for a sensitive operation.
        """
        # Simple path matching for this POC.
        return any(request.url.path.startswith(path) for path in SENSITIVE_PATHS)

    def _has_recent_mfa(self, token: str | None) -> bool:
        """
        Placeholder for checking if the user's JWT indicates recent MFA.

        In a real system, this function would:
        1. Decode the JWT.
        2. Check for a specific 'mfa_time' or 'amr' (Authentication Methods Reference) claim.
        3. Verify that the timestamp is within a defined window (e.g., last 15 minutes).
        """
        # For this POC, we will simply return False to simulate the need for a step-up.
        # To test the "passing" case, you could change this to `return True`.
        if not token:
            return False # No token, no recent MFA.

        # In a real implementation, you would decode the token and check its claims.
        # For now, we simulate the failure case.
        return False