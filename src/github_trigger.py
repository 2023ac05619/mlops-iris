import os
import json
import httpx
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GitHubActionsTrigger:
    """Service to trigger GitHub Actions workflows programmatically."""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY')  # format: "owner/repo"
        self.workflow_file = os.getenv('GITHUB_WORKFLOW_FILE', 'ci-cd.yml')
        self.base_url = "https://api.github.com"
        
        # Validate configuration
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not set - GitHub Actions triggering will not work")
        if not self.github_repo:
            logger.warning("GITHUB_REPOSITORY not set - GitHub Actions triggering will not work")
    
    def is_configured(self) -> bool:
        """Check if GitHub integration is properly configured."""
        return bool(self.github_token and self.github_repo)
    
    async def trigger_retraining_workflow(self, reason: str = "manual", 
                                        additional_inputs: Dict[str, str] = None) -> bool:
        """Trigger the CI/CD workflow for retraining."""
        if not self.is_configured():
            logger.error("GitHub Actions integration not configured")
            return False
        
        try:
            # Prepare workflow inputs
            workflow_inputs = {
                "retrain_trigger": "true",
                "trigger_reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
            if additional_inputs:
                workflow_inputs.update(additional_inputs)
            
            # Trigger workflow
            success = await self._trigger_workflow(
                workflow_id=self.workflow_file,
                ref="main",
                inputs=workflow_inputs
            )
            
            if success:
                logger.info(f"Successfully triggered retraining workflow. Reason: {reason}")
            else:
                logger.error(f"Failed to trigger retraining workflow. Reason: {reason}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error triggering retraining workflow: {e}")
            return False
    
    async def _trigger_workflow(self, workflow_id: str, ref: str = "main", 
                              inputs: Dict[str, str] = None) -> bool:
        """Trigger a GitHub Actions workflow via API."""
        try:
            url = f"{self.base_url}/repos/{self.github_repo}/actions/workflows/{workflow_id}/dispatches"
            
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"
            }
            
            payload = {
                "ref": ref
            }
            
            if inputs:
                payload["inputs"] = inputs
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 204:
                    logger.info(f"Workflow {workflow_id} triggered successfully")
                    return True
                else:
                    logger.error(f"Failed to trigger workflow. Status: {response.status_code}, Response: {response.text}")
                    return False
                    
        except httpx.TimeoutException:
            logger.error("Timeout while triggering GitHub workflow")
            return False
        except Exception as e:
            logger.error(f"Error triggering GitHub workflow: {e}")
            return False
    
    async def get_workflow_runs(self, workflow_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent workflow runs for monitoring."""
        if not self.is_configured():
            return {"error": "GitHub integration not configured"}
        
        try:
            url = f"{self.base_url}/repos/{self.github_repo}/actions/workflows/{workflow_id}/runs"
            
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            params = {
                "per_page": limit,
                "page": 1
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "workflow_runs": data.get("workflow_runs", []),
                        "total_count": data.get("total_count", 0)
                    }
                else:
                    logger.error(f"Failed to get workflow runs. Status: {response.status_code}")
                    return {"error": f"API request failed with status {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error getting workflow runs: {e}")
            return {"error": str(e)}
    
    async def get_latest_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of the latest workflow run."""
        try:
            runs_data = await self.get_workflow_runs(workflow_id, limit=1)
            
            if "error" in runs_data:
                return runs_data
            
            workflow_runs = runs_data.get("workflow_runs", [])
            
            if not workflow_runs:
                return {"status": "no_runs", "message": "No workflow runs found"}
            
            latest_run = workflow_runs[0]
            
            return {
                "run_id": latest_run.get("id"),
                "status": latest_run.get("status"),
                "conclusion": latest_run.get("conclusion"),
                "created_at": latest_run.get("created_at"),
                "updated_at": latest_run.get("updated_at"),
                "html_url": latest_run.get("html_url"),
                "head_branch": latest_run.get("head_branch"),
                "event": latest_run.get("event"),
                "actor": latest_run.get("actor", {}).get("login")
            }
            
        except Exception as e:
            logger.error(f"Error getting latest workflow status: {e}")
            return {"error": str(e)}
    
    def trigger_retraining_workflow_sync(self, reason: str = "manual", 
                                       additional_inputs: Dict[str, str] = None) -> bool:
        """Synchronous version of trigger_retraining_workflow for use in background tasks."""
        import asyncio
        
        try:
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                self.trigger_retraining_workflow(reason, additional_inputs)
            )
            
        except Exception as e:
            logger.error(f"Error in synchronous workflow trigger: {e}")
            return False
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get the current configuration status for debugging."""
        return {
            "is_configured": self.is_configured(),
            "has_github_token": bool(self.github_token),
            "github_repo": self.github_repo,
            "workflow_file": self.workflow_file,
            "github_token_length": len(self.github_token) if self.github_token else 0,
            "base_url": self.base_url
        }


class WebhookHandler:
    """Handle GitHub webhook events for workflow status updates."""
    
    def __init__(self, webhook_secret: Optional[str] = None):
        self.webhook_secret = webhook_secret or os.getenv('GITHUB_WEBHOOK_SECRET')
    
    def verify_signature(self, payload_body: bytes, signature_header: str) -> bool:
        """Verify GitHub webhook signature."""
        if not self.webhook_secret:
            logger.warning("No webhook secret configured - skipping signature verification")
            return True
        
        import hmac
        import hashlib
        
        try:
            signature = signature_header.split('=')[1]
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload_body,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False
    
    def handle_workflow_run_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow_run webhook event."""
        try:
            action = payload.get('action')
            workflow_run = payload.get('workflow_run', {})
            
            event_data = {
                "event_type": "workflow_run",
                "action": action,
                "workflow_name": workflow_run.get('name'),
                "workflow_id": workflow_run.get('workflow_id'),
                "run_id": workflow_run.get('id'),
                "status": workflow_run.get('status'),
                "conclusion": workflow_run.get('conclusion'),
                "created_at": workflow_run.get('created_at'),
                "updated_at": workflow_run.get('updated_at'),
                "head_branch": workflow_run.get('head_branch'),
                "event": workflow_run.get('event'),
                "actor": workflow_run.get('actor', {}).get('login'),
                "repository": payload.get('repository', {}).get('full_name')
            }
            
            # Log significant events
            if action in ['completed', 'requested', 'in_progress']:
                logger.info(f"Workflow {workflow_run.get('name')} {action}: {workflow_run.get('conclusion')}")
            
            return event_data
            
        except Exception as e:
            logger.error(f"Error handling workflow run event: {e}")
            return {"error": str(e)}
    
    def handle_repository_dispatch_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repository_dispatch webhook event."""
        try:
            event_type = payload.get('action')
            client_payload = payload.get('client_payload', {})
            
            event_data = {
                "event_type": "repository_dispatch",
                "action": event_type,
                "client_payload": client_payload,
                "repository": payload.get('repository', {}).get('full_name'),
                "sender": payload.get('sender', {}).get('login')
            }
            
            logger.info(f"Repository dispatch event: {event_type}")
            
            return event_data
            
        except Exception as e:
            logger.error(f"Error handling repository dispatch event: {e}")
            return {"error": str(e)}