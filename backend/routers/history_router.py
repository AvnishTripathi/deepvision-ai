import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

import schemas, models, auth
from database import get_db

router = APIRouter(prefix="/api/v1", tags=["history"])

@router.get("/history", response_model=schemas.HistoryResponse)
def get_history(
    page: int = 1,
    limit: int = 10,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    skip = (page - 1) * limit
    
    query = db.query(models.DetectionJob).filter(models.DetectionJob.user_id == current_user.id)
    total = query.count()
    
    jobs = query.order_by(models.DetectionJob.created_at.desc()).offset(skip).limit(limit).all()
    
    items = []
    for job in jobs:
        overall_score = None
        label = None
        if job.result_json and job.status == "done":
            try:
                res = json.loads(job.result_json)
                overall_score = res.get("overall_score")
                label = res.get("label")
            except:
                pass
                
        items.append(schemas.HistoryItem(
            id=job.id,
            type=job.type,
            status=job.status,
            original_filename=job.original_filename,
            created_at=job.created_at,
            completed_at=job.completed_at,
            overall_score=overall_score,
            label=label
        ))
        
    return schemas.HistoryResponse(
        items=items,
        total=total,
        page=page,
        limit=limit
    )

@router.delete("/history/{job_id}")
def delete_history_item(
    job_id: str,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    job = db.query(models.DetectionJob).filter(
        models.DetectionJob.id == job_id,
        models.DetectionJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Item not found")
        
    db.delete(job)
    db.commit()
    
    return {"detail": "Item deleted"}

@router.get("/stats")
def get_stats(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    jobs = db.query(models.DetectionJob).filter(
        models.DetectionJob.user_id == current_user.id,
        models.DetectionJob.status == "done"
    ).all()
    
    total_scans = len(jobs)
    deepfakes_found = 0
    total_score = 0
    scored_scans = 0
    
    for job in jobs:
        if job.result_json:
            try:
                res = json.loads(job.result_json)
                if res.get("label") == "fake":
                    deepfakes_found += 1
                if "overall_score" in res:
                    total_score += res["overall_score"]
                    scored_scans += 1
            except:
                pass
                
    avg_score = total_score / scored_scans if scored_scans > 0 else 0
    
    return {
        "total_scans": total_scans,
        "deepfakes_found": deepfakes_found,
        "avg_score": round(avg_score, 4)
    }
