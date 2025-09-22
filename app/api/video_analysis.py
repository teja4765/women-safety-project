"""
Video analysis API endpoints for file upload and batch processing
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import logging
import os
import uuid
from datetime import datetime
import json

from app.core.database import get_db
from app.services.video_file_processor import VideoFileProcessor
from app.schemas.video_analysis import VideoAnalysisJob, VideoAnalysisResponse, VideoAnalysisStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Global video file processor instance
video_processor = None

async def get_video_processor():
    """Get video file processor instance"""
    global video_processor
    if video_processor is None:
        video_processor = VideoFileProcessor()
        await video_processor.initialize()
    return video_processor


@router.post("/video-analysis/upload", response_model=VideoAnalysisJob)
async def upload_video_for_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to analyze"),
    zone_id: str = Form(..., description="Zone ID for analysis context"),
    camera_id: Optional[str] = Form(None, description="Camera ID (optional)"),
    processing_mode: str = Form("batch", description="Processing mode: batch or realtime"),
    db: Session = Depends(get_db)
):
    """Upload video file for safety analysis"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create upload directory
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.mp4'
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join(upload_dir, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get video processor
        processor = await get_video_processor()
        
        # Start processing job
        job_info = await processor.process_video_file(
            video_path=file_path,
            zone_id=zone_id,
            camera_id=camera_id,
            processing_mode=processing_mode
        )
        
        logger.info(f"Video upload and analysis started: {job_info['job_id']}")
        
        return VideoAnalysisJob(
            job_id=job_info["job_id"],
            filename=file.filename,
            zone_id=zone_id,
            camera_id=job_info["camera_id"],
            processing_mode=processing_mode,
            status=job_info["status"],
            progress=job_info["progress"],
            created_at=job_info["start_time"]
        )
        
    except Exception as e:
        logger.error(f"Error uploading video for analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@router.post("/video-analysis/process-file", response_model=VideoAnalysisJob)
async def process_existing_video_file(
    background_tasks: BackgroundTasks,
    video_path: str = Form(..., description="Path to existing video file"),
    zone_id: str = Form(..., description="Zone ID for analysis context"),
    camera_id: Optional[str] = Form(None, description="Camera ID (optional)"),
    processing_mode: str = Form("batch", description="Processing mode: batch or realtime"),
    db: Session = Depends(get_db)
):
    """Process an existing video file for safety analysis"""
    try:
        # Validate file exists
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Get video processor
        processor = await get_video_processor()
        
        # Start processing job
        job_info = await processor.process_video_file(
            video_path=video_path,
            zone_id=zone_id,
            camera_id=camera_id,
            processing_mode=processing_mode
        )
        
        logger.info(f"Video file analysis started: {job_info['job_id']}")
        
        return VideoAnalysisJob(
            job_id=job_info["job_id"],
            filename=os.path.basename(video_path),
            zone_id=zone_id,
            camera_id=job_info["camera_id"],
            processing_mode=processing_mode,
            status=job_info["status"],
            progress=job_info["progress"],
            created_at=job_info["start_time"]
        )
        
    except Exception as e:
        logger.error(f"Error processing video file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@router.get("/video-analysis/jobs", response_model=List[VideoAnalysisResponse])
async def get_video_analysis_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    db: Session = Depends(get_db)
):
    """Get list of video analysis jobs"""
    try:
        processor = await get_video_processor()
        all_jobs = processor.get_all_jobs()
        
        # Combine active and completed jobs
        jobs = []
        jobs.extend(all_jobs.get("active", {}).values())
        jobs.extend(all_jobs.get("completed", {}).values())
        
        # Filter by status if specified
        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.get("start_time", datetime.min), reverse=True)
        
        # Limit results
        jobs = jobs[:limit]
        
        # Convert to response format
        response_jobs = []
        for job in jobs:
            response_jobs.append(VideoAnalysisResponse(
                job_id=job["job_id"],
                filename=os.path.basename(job.get("video_path", "")),
                zone_id=job["zone_id"],
                camera_id=job["camera_id"],
                processing_mode=job["processing_mode"],
                status=job["status"],
                progress=job.get("progress", 0.0),
                created_at=job["start_time"],
                completed_at=job.get("end_time"),
                total_frames=job.get("total_frames", 0),
                processed_frames=job.get("processed_frames", 0),
                alerts_found=job.get("alerts_found", 0),
                errors=job.get("errors", [])
            ))
        
        return response_jobs
        
    except Exception as e:
        logger.error(f"Error getting video analysis jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/video-analysis/jobs/{job_id}", response_model=VideoAnalysisResponse)
async def get_video_analysis_job(job_id: str, db: Session = Depends(get_db)):
    """Get specific video analysis job details"""
    try:
        processor = await get_video_processor()
        job_info = processor.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return VideoAnalysisResponse(
            job_id=job_info["job_id"],
            filename=os.path.basename(job_info.get("video_path", "")),
            zone_id=job_info["zone_id"],
            camera_id=job_info["camera_id"],
            processing_mode=job_info["processing_mode"],
            status=job_info["status"],
            progress=job_info.get("progress", 0.0),
            created_at=job_info["start_time"],
            completed_at=job_info.get("end_time"),
            total_frames=job_info.get("total_frames", 0),
            processed_frames=job_info.get("processed_frames", 0),
            alerts_found=job_info.get("alerts_found", 0),
            errors=job_info.get("errors", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video analysis job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/video-analysis/jobs/{job_id}")
async def cancel_video_analysis_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a video analysis job"""
    try:
        processor = await get_video_processor()
        success = processor.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or already completed")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling video analysis job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/video-analysis/jobs/{job_id}/results")
async def get_video_analysis_results(job_id: str, db: Session = Depends(get_db)):
    """Get results from a completed video analysis job"""
    try:
        processor = await get_video_processor()
        job_info = processor.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_info["status"] not in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        # Get results directory
        output_dir = job_info.get("output_dir")
        if not output_dir or not os.path.exists(output_dir):
            raise HTTPException(status_code=404, detail="Results not found")
        
        # Get processing report
        report_path = os.path.join(output_dir, "processing_report.json")
        report = {}
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
        # Merge detection stats from job_info if present
        detection_stats = job_info.get("detection_stats")
        if detection_stats:
            report["detection_stats"] = detection_stats
        # Include violence stats
        violence_stats = job_info.get("violence_stats")
        if violence_stats:
            report["violence_stats"] = violence_stats
        
        # Get alerts
        alerts_dir = os.path.join(output_dir, "alerts")
        alerts = []
        if os.path.exists(alerts_dir):
            for alert_file in os.listdir(alerts_dir):
                if alert_file.endswith('.json'):
                    alert_path = os.path.join(alerts_dir, alert_file)
                    with open(alert_path, 'r') as f:
                        alert_data = json.load(f)
                        alerts.append(alert_data)
        
        # Get annotated frames (sample)
        frame_files = []
        for file in os.listdir(output_dir):
            if file.startswith('frame_') and file.endswith('.jpg'):
                frame_files.append(file)
        
        # Sort and take first 10 frames as sample
        frame_files.sort()
        sample_frames = frame_files[:10]
        
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "processing_report": report,
            "alerts": alerts,
            "sample_frames": sample_frames,
            "output_directory": output_dir,
            "total_alerts": len(alerts),
            "total_frames": len(frame_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video analysis results {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/video-analysis/jobs/{job_id}/download")
async def download_video_analysis_results(job_id: str, db: Session = Depends(get_db)):
    """Download results from a completed video analysis job as a zip file"""
    try:
        processor = await get_video_processor()
        job_info = processor.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_info["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        # Get results directory
        output_dir = job_info.get("output_dir")
        if not output_dir or not os.path.exists(output_dir):
            raise HTTPException(status_code=404, detail="Results not found")
        
        # Create zip file
        import zipfile
        import tempfile
        
        zip_filename = f"video_analysis_{job_id}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        
        # Return file for download
        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type='application/zip'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video analysis results {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/video-analysis/jobs/{job_id}/frames/{filename}")
async def get_video_analysis_frame(job_id: str, filename: str, db: Session = Depends(get_db)):
    """Serve a generated annotated frame for a job"""
    try:
        processor = await get_video_processor()
        job_info = processor.get_job_status(job_id)

        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")

        output_dir = job_info.get("output_dir")
        if not output_dir or not os.path.exists(output_dir):
            raise HTTPException(status_code=404, detail="Output directory not found")

        frame_path = os.path.join(output_dir, filename)
        if not os.path.exists(frame_path):
            raise HTTPException(status_code=404, detail="Frame not found")

        return FileResponse(frame_path, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving frame {filename} for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/video-analysis/status")
async def get_video_analysis_system_status(db: Session = Depends(get_db)):
    """Get video analysis system status"""
    try:
        processor = await get_video_processor()
        all_jobs = processor.get_all_jobs()
        
        active_jobs = len(all_jobs.get("active", {}))
        completed_jobs = len(all_jobs.get("completed", {}))
        
        return {
            "status": "operational",
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "total_jobs": active_jobs + completed_jobs,
            "processor_initialized": processor is not None
        }
        
    except Exception as e:
        logger.error(f"Error getting video analysis system status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
