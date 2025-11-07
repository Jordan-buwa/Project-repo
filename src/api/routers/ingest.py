from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.api.utils.database import get_db_connection, save_customer_data, save_batch_customer_data, generate_batch_id
from src.api.utils.customer_data import CustomerData, BatchCustomerData
from fastapi.responses import PlainTextResponse
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import logging
from datetime import datetime
import io

router = APIRouter(prefix="/ingest")
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='src/api/utils/logs/ingest.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)


class IngestResponse(BaseModel):
    """Response model for data ingestion"""
    success: bool
    message: str
    records_processed: int
    records_saved: int
    records_failed: int
    failed_records: Optional[List[Dict]] = None
    batch_id: Optional[str] = None

def parse_csv_file(file_content: bytes) -> pd.DataFrame:
    """Parse CSV file content"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")

def parse_excel_file(file_content: bytes) -> pd.DataFrame:
    """Parse Excel file content"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise ValueError(f"Error parsing Excel file: {str(e)}")
    
def log_ingestion(batch_id: str, source: str, processed: int, saved: int, 
                  failed: int, file_name: Optional[str] = None, 
                  status: str = "success", error: Optional[str] = None):
    """Log ingestion activity"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ingestion_logs (
                    batch_id, source, records_processed, records_saved,
                    records_failed, file_name, status, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (batch_id) DO UPDATE SET
                    records_processed = EXCLUDED.records_processed,
                    records_saved = EXCLUDED.records_saved,
                    records_failed = EXCLUDED.records_failed,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message
            """, (batch_id, source, processed, saved, failed, file_name, status, error))
            cursor.close()
    except Exception as e:
        logger.error(f"Error logging ingestion: {str(e)}")

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names - handle common variations"""
    # Normalize to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Handle common column name variations
    column_mapping = {
        'customerid': 'customer_id',
        'customer id': 'customer_id',
        'cust_id': 'customer_id'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that should not be stored (from config)"""
    drop_cols = ["Unnamed: 0", "X", "customer", "traintest", "churndep"]
    
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"Dropped column: {col}")
    
    return df

def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert churn column to boolean if needed"""
    if 'churn' in df.columns:
        # Handle various representations
        df['churn'] = df['churn'].map({
            'Yes': 1, 'yes': 1, 'YES': 1, 'Y': 1, 'y': 1, '1': 1, True: 1,
            'No': 0, 'no': 0, 'NO': 0, 'N': 0, 'n': 0, '0': 0, False: 0
        })
    
    return df

def dataframe_to_customer_data(df: pd.DataFrame, source: str) -> List[CustomerData]:
    """Convert DataFrame to list of CustomerData objects"""
    customers = []
    
    for idx, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            row_dict = {k: (v if pd.notna(v) else None) for k, v in row_dict.items()}
            
            row_dict['source'] = source
            row_dict['timestamp'] = datetime.utcnow().isoformat()
            
            customer = CustomerData(**row_dict)
            customers.append(customer)
        
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {str(e)}")
            continue
    
    return customers

@router.post("/single", response_model=IngestResponse)
async def ingest_single_record(data: CustomerData):
    """
    Ingest a single customer record via POST request.
    
    Args:
        data: Customer data object
    
    Returns:
        Ingestion response with status
    """
    batch_id = generate_batch_id()
    
    try:
        data.source = "api"
        data.timestamp = datetime.utcnow().isoformat()
        
        success = save_customer_data(data, batch_id)
        
        if success:
            log_ingestion(batch_id, "api", 1, 1, 0)
            logger.info(f"Successfully ingested single record: {data.customer_id}")
            
            return IngestResponse(
                success=True,
                message="Record ingested successfully",
                records_processed=1,
                records_saved=1,
                records_failed=0,
                batch_id=batch_id
            )
        else:
            log_ingestion(batch_id, "api", 1, 0, 1, error="Failed to save record")
            raise HTTPException(status_code=500, detail="Failed to save record")
    
    except Exception as e:
        logger.error(f"Error ingesting single record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=IngestResponse)
async def ingest_batch_records(data: BatchCustomerData):
    """
    Ingest multiple customer records via POST request.
    Uses bulk insert for better performance.
    
    Args:
        data: Batch customer data object
    
    Returns:
        Ingestion response with statistics
    """
    batch_id = generate_batch_id()
    
    try:
        # Set metadata for all customers
        for customer in data.customers:
            customer.source = "api_batch"
            customer.timestamp = datetime.utcnow().isoformat()
        
        # Use bulk insert
        saved, failed = save_batch_customer_data(data.customers, batch_id)
        processed = len(data.customers)
        
        log_ingestion(batch_id, "api_batch", processed, saved, failed)
        logger.info(f"Batch ingestion completed: {saved}/{processed} records saved")
        
        return IngestResponse(
            success=True,
            message=f"Batch ingestion completed",
            records_processed=processed,
            records_saved=saved,
            records_failed=failed,
            batch_id=batch_id
        )
    
    except Exception as e:
        logger.error(f"Error in batch ingestion: {str(e)}")
        log_ingestion(batch_id, "api_batch", 0, 0, len(data.customers), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/csv", response_model=IngestResponse)
async def ingest_csv_file(
    file: UploadFile = File(..., description="CSV file to ingest"),
    source: str = Form(default="csv", description="Data source identifier")
):
    """
    Ingest customer data from CSV file.
    Supports Google Forms CSV output.
    
    Args:
        file: CSV file upload
        source: Source identifier (e.g., 'google_form', 'manual_csv')
    
    Returns:
        Ingestion response with statistics
    """
    batch_id = generate_batch_id()
    processed = 0
    saved = 0
    failed = 0
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        content = await file.read()
        
        df = parse_csv_file(content)
        logger.info(f"Parsed CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        df = normalize_column_names(df)
        df = convert_boolean_columns(df)
        
        customers = dataframe_to_customer_data(df, source)
        processed = len(df)
        
        # Use bulk insert for better performance
        saved, failed = save_batch_customer_data(customers, batch_id)
        
        log_ingestion(batch_id, source, processed, saved, failed, file_name=file.filename)
        logger.info(f"CSV ingestion completed: {saved}/{processed} records saved from {file.filename}")
        
        return IngestResponse(
            success=True,
            message=f"CSV file processed successfully",
            records_processed=processed,
            records_saved=saved,
            records_failed=failed,
            batch_id=batch_id
        )
    
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        log_ingestion(batch_id, source, processed, saved, failed, 
                     file_name=file.filename, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@router.post("/excel", response_model=IngestResponse)
async def ingest_excel_file(
    file: UploadFile = File(..., description="Excel file to ingest"),
    source: str = Form(default="excel", description="Data source identifier")
):
    """
    Ingest customer data from Excel file (.xlsx, .xls).
    
    Args:
        file: Excel file upload
        source: Source identifier
    
    Returns:
        Ingestion response with statistics
    """
    batch_id = generate_batch_id()
    processed = 0
    saved = 0
    failed = 0
    
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
    try:
        content = await file.read()
        
        df = parse_excel_file(content, file.filename)
        logger.info(f"Parsed Excel with {len(df)} rows and columns: {list(df.columns)}")
        
        df = normalize_column_names(df)
        df = convert_boolean_columns(df)
        
        customers = dataframe_to_customer_data(df, source)
        processed = len(df)
        
        # Use bulk insert
        saved, failed = save_batch_customer_data(customers, batch_id)
        
        log_ingestion(batch_id, source, processed, saved, failed, file_name=file.filename)
        logger.info(f"Excel ingestion completed: {saved}/{processed} records saved from {file.filename}")
        
        return IngestResponse(
            success=True,
            message=f"Excel file processed successfully",
            records_processed=processed,
            records_saved=saved,
            records_failed=failed,
            batch_id=batch_id
        )
    
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        log_ingestion(batch_id, source, processed, saved, failed,
                     file_name=file.filename, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing Excel: {str(e)}")

@router.get("/stats")
async def get_ingestion_stats(limit: int = 10):
    """
    Get ingestion statistics and recent logs.
    
    Args:
        limit: Number of recent logs to return
    
    Returns:
        Ingestion statistics
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total records
            cursor.execute("SELECT COUNT(*) as total FROM customer_data")
            total_records = cursor.fetchone()['total']
            
            # Get records by source
            cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM customer_data 
                GROUP BY source
            """)
            by_source = {row['source']: row['count'] for row in cursor.fetchall()}
            
            # Get recent ingestion logs
            cursor.execute("""
                SELECT * FROM ingestion_logs 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (limit,))
            recent_logs = [dict(row) for row in cursor.fetchall()]
            
            # Get ingestion summary
            cursor.execute("""
                SELECT 
                    SUM(records_processed) as total_processed,
                    SUM(records_saved) as total_saved,
                    SUM(records_failed) as total_failed
                FROM ingestion_logs
            """)
            summary = dict(cursor.fetchone())
            
            cursor.close()
            
            return {
                "total_records": total_records,
                "records_by_source": by_source,
                "ingestion_summary": summary,
                "recent_logs": recent_logs
            }
    
    except Exception as e:
        logger.error(f"Error getting ingestion stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch/{batch_id}")
async def get_batch_details(batch_id: str):
    """
    Get details of a specific ingestion batch.
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        Batch details including records
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM ingestion_logs 
                WHERE batch_id = %s
            """, (batch_id,))
            log = cursor.fetchone()
            
            if not log:
                raise HTTPException(status_code=404, detail="Batch not found")
            
            cursor.execute("""
                SELECT * FROM customer_data 
                WHERE batch_id = %s
                ORDER BY created_at
            """, (batch_id,))
            records = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            
            return {
                "batch_info": dict(log),
                "records": records,
                "record_count": len(records)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str):
    """
    Delete a specific ingestion batch.
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        Deletion confirmation
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM customer_data WHERE batch_id = %s", (batch_id,))
            deleted_records = cursor.rowcount
            
            cursor.execute("DELETE FROM ingestion_logs WHERE batch_id = %s", (batch_id,))
            
            cursor.close()
            
            if deleted_records == 0:
                raise HTTPException(status_code=404, detail="Batch not found")
            
            logger.info(f"Deleted batch {batch_id} with {deleted_records} records")
            
            return {
                "success": True,
                "message": f"Deleted batch {batch_id}",
                "records_deleted": deleted_records
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export")
async def export_data(
    format: str = "csv",
    source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Export ingested data.
    
    Args:
        format: Export format (csv or json)
        source: Filter by source
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
    
    Returns:
        Exported data
    """
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM customer_data WHERE 1=1"
            params = []
            
            if source:
                query += " AND source = %s"
                params.append(source)
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " ORDER BY created_at DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if format.lower() == "csv":
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                return {
                    "format": "csv",
                    "data": csv_buffer.getvalue(),
                    "record_count": len(df)
                }
            else:
                return {
                    "format": "json",
                    "data": df.to_dict(orient='records'),
                    "record_count": len(df)
                }
    
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers")

async def get_customers(limit: int = 100, offset: int = 0):

    try:

        with get_db_connection() as conn:

            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(

                "SELECT * FROM customer_data ORDER BY created_at DESC LIMIT %s OFFSET %s",

                (limit, offset)

            )

            rows = [dict(r) for r in cur.fetchall()]

            cur.close()

            return {"records": rows, "count": len(rows)}

    except Exception as e:

        logger.error(f"Error fetching customers: {e}")

        raise HTTPException(status_code=500, detail=str(e))
