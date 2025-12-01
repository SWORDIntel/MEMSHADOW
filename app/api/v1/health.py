from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "ok", "version": settings.VERSION}