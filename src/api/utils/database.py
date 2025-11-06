import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from datetime import datetime
from typing import List
from src.api.utils.customer_data import CustomerData
import json
import os
import logging
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='src/api/utils/logs/database.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)
# PostgreSQL Configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB_NAME', 'churn_db'),
    'user': os.getenv('POSTGRES_DB_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'sslmode': 'require'
}

# Connection pool (min 2 connections, max 10)
connection_pool = None

def initialize_connection_pool():
    """Initialize PostgreSQL connection pool"""
    global connection_pool
    try:
        connection_pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **DB_CONFIG
        )
        logger.info("PostgreSQL connection pool created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating connection pool: {str(e)}")
        return False

if not initialize_connection_pool():
    raise RuntimeError(
        "Database connection pool failed to initialize. "
    )

@contextmanager
def get_db_connection():
    """Context manager for database connections from pool"""
    conn = connection_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        connection_pool.putconn(conn)

# Database initialization function
def initialize_database():
    """Initialize the database schema with your features"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Main customer data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_data (
                 features JSONB,      
                id SERIAL PRIMARY KEY,
                customer_id VARCHAR(255),
                       
                -- Columns to drop in preprocessing       
                unnamed_0 INTEGER,
                x INTEGER,                     
                customer VARCHAR(255), 
                traintest VARCHAR(10),    
                churndep VARCHAR(10),          
                
                -- Numerical Features
                revenue DECIMAL(10, 2),
                mou DECIMAL(10, 2),
                recchrge DECIMAL(10, 2),
                directas DECIMAL(10, 2),
                overage DECIMAL(10, 2),
                roam DECIMAL(10, 2),
                changem DECIMAL(10, 2),
                changer DECIMAL(10, 2),
                dropvce DECIMAL(10, 2),
                blckvce DECIMAL(10, 2),
                unansvce DECIMAL(10, 2),
                custcare DECIMAL(10, 2),
                threeway DECIMAL(10, 2),
                mourec DECIMAL(10, 2),
                outcalls DECIMAL(10, 2),
                incalls DECIMAL(10, 2),
                peakvce DECIMAL(10, 2),
                opeakvce DECIMAL(10, 2),
                dropblk DECIMAL(10, 2),
                callfwdv DECIMAL(10, 2),
                callwait DECIMAL(10, 2),
                months DECIMAL(10, 2),
                uniqsubs DECIMAL(10, 2),
                actvsubs DECIMAL(10, 2),
                phones DECIMAL(10, 2),
                models DECIMAL(10, 2),
                eqpdays DECIMAL(10, 2),
                age1 DECIMAL(10, 2),
                age2 DECIMAL(10, 2),
                refer DECIMAL(10, 2),
                income DECIMAL(10, 2),
                setprc DECIMAL(10, 2),
                
                -- Categorical Features (stored as VARCHAR)
                children VARCHAR(10),
                credita VARCHAR(10),
                creditaa VARCHAR(10),
                prizmrur VARCHAR(10),
                prizmub VARCHAR(10),
                prizmtwn VARCHAR(10),
                refurb VARCHAR(10),
                webcap VARCHAR(10),
                truck VARCHAR(10),
                rv VARCHAR(10),
                occprof VARCHAR(10),
                occcler VARCHAR(10),
                occcrft VARCHAR(10),
                occstud VARCHAR(10),
                occhmkr VARCHAR(10),
                occret VARCHAR(10),
                occself VARCHAR(10),
                ownrent VARCHAR(10),
                marryun VARCHAR(10),
                marryyes VARCHAR(10),
                mailord VARCHAR(10),
                mailres VARCHAR(10),
                mailflag VARCHAR(10),
                travel VARCHAR(10),
                pcown VARCHAR(10),
                creditcd VARCHAR(10),
                newcelly VARCHAR(10),
                newcelln VARCHAR(10),
                incmiss VARCHAR(10),
                mcycle VARCHAR(10),
                setprcm VARCHAR(10),
                retcall VARCHAR(10),
                retcalls VARCHAR(10),
                retaccpt VARCHAR(10),
                
                -- Target Variable
                churn BOOLEAN,
                
                -- Metadata
                source VARCHAR(100),
                timestamp TIMESTAMP,
                batch_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                CONSTRAINT unique_customer_timestamp UNIQUE(customer_id, timestamp)
            )
        """)
        
        # Ingestion logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_logs (
                id SERIAL PRIMARY KEY,
                batch_id VARCHAR(255) UNIQUE,
                source VARCHAR(100),
                records_processed INTEGER,
                records_saved INTEGER,
                records_failed INTEGER,
                file_name VARCHAR(500),
                status VARCHAR(50),
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for commonly queried fields
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON customer_data(customer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON customer_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_id ON customer_data(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON customer_data(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_churn ON customer_data(churn)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_revenue ON customer_data(revenue)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_months ON customer_data(months)")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("PostgreSQL database initialized successfully with custom schema")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

# Initialize connection pool and database
def startup():
    """Startup function to initialize database"""
    if initialize_connection_pool():
        initialize_database()
    else:
        logger.error("Failed to initialize connection pool")

# Call startup
startup()

def generate_batch_id() -> str:
    """Generate unique batch ID"""
    return f"BATCH_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(object())}"

def save_customer_data(data: CustomerData, batch_id: str) -> bool:
    """Save a single customer record to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Generate customer_id if not provided
            if not data.customer_id:
                data.customer_id = f"CUST_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(data)}"

            # Convert to features JSON
            features_dict = data.dict(exclude={'customer_id', 'source', 'timestamp', 'batch_id'})
            features_json = json.dumps(features_dict)
            cursor.execute("""
                INSERT INTO customer_data (
                    customer_id, unnamed_0, x, customer, traintest, churndep, revenue, mou, recchrge,
                    directas, overage, roam, changem, changer, dropvce, blckvce,
                    unansvce, custcare, threeway,mourec, outcalls, incalls, peakvce,
                    opeakvce, dropblk, callfwdv,callwait, months, uniqsubs, actvsubs,
                    phones, models, eqpdays, age1, age2, refer, income, setprc,
                    children, credita, creditaa, prizmrur, prizmub, prizmtwn, refurb,
                    webcap, truck, rv, occprof, occcler, occcrft, occstud, occhmkr,
                    occret, occself, ownrent, marryun, marryyes, mailord, mailres,
                    mailflag, travel, pcown, creditcd, newcelly, newcelln, incmiss,
                    mcycle, setprcm, retcall, retcalls, retaccpt,
                    churn, source, timestamp, batch_id, features
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (customer_id, timestamp) DO UPDATE SET
                    revenue = EXCLUDED.revenue,
                    mou = EXCLUDED.mou,
                    churn = EXCLUDED.churn,
                    features = EXCLUDED.features
            """, (
                data.customer_id, data.unnamed_0, data.x, data.customer, data.traintest, data.churndep,
                data.revenue, data.mou, data.recchrge, data.directas,
                data.overage, data.roam, data.changem, data.changer, data.dropvce,
                data.blckvce, data.unansvce, data.custcare, data.threeway, data.mourec,
                data.outcalls, data.incalls, data.peakvce, data.opeakvce, data.dropblk,
                data.callfwdv, data.callwait, data.months, data.uniqsubs, data.actvsubs,
                data.phones, data.models, data.eqpdays, data.age1, data.age2, data.refer,
                data.income, data.setprc,
                data.children, data.credita, data.creditaa, data.prizmrur, data.prizmub,
                data.prizmtwn, data.refurb, data.webcap, data.truck, data.rv, data.occprof,
                data.occcler, data.occcrft, data.occstud, data.occhmkr, data.occret,
                data.occself, data.ownrent, data.marryun, data.marryyes, data.mailord,
                data.mailres, data.mailflag, data.travel, data.pcown, data.creditcd,
                data.newcelly, data.newcelln, data.incmiss, data.mcycle, data.setprcm,
                data.retcall, data.retcalls, data.retaccpt,
                data.churn, data.source, data.timestamp, batch_id,
                json.dump(features_dict)
            ))
            
            cursor.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving customer data: {str(e)}")
        return False

def save_batch_customer_data(customers: List[CustomerData], batch_id: str) -> tuple:
    """
    Save multiple customer records efficiently using execute_values
    Returns: (saved_count, failed_count)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare data for bulk insert
            values = []
            for customer in customers:
                # Generate customer_id if not provided
                if not customer.customer_id:
                    customer.customer_id = f"CUST_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(customer)}"
                
                values.append((
                    customer.customer_id, customer.unnamed_0, customer.x, customer.customer, customer.traintest, customer.churndep, customer.revenue, customer.mou, customer.recchrge,
                    customer.directas, customer.overage, customer.roam, customer.changem,
                    customer.changer, customer.dropvce, customer.blckvce, customer.unansvce,
                    customer.custcare, customer.threeway, customer.mourec, customer.outcalls,
                    customer.incalls, customer.peakvce, customer.opeakvce, customer.dropblk,
                    customer.callfwdv, customer.callwait, customer.months, customer.uniqsubs,
                    customer.actvsubs, customer.phones, customer.models, customer.eqpdays,
                    customer.age1, customer.age2, customer.refer, customer.income, customer.setprc,
                    customer.children, customer.credita, customer.creditaa, customer.prizmrur,
                    customer.prizmub, customer.prizmtwn, customer.refurb, customer.webcap,
                    customer.truck, customer.rv, customer.occprof, customer.occcler,
                    customer.occcrft, customer.occstud, customer.occhmkr, customer.occret,
                    customer.occself, customer.ownrent, customer.marryun, customer.marryyes,
                    customer.mailord, customer.mailres, customer.mailflag, customer.travel,
                    customer.pcown, customer.creditcd, customer.newcelly, customer.newcelln,
                    customer.incmiss, customer.mcycle, customer.setprcm, customer.retcall,
                    customer.retcalls, customer.retaccpt,
                    customer.churn, customer.source, customer.timestamp, batch_id
                ))
            
            # Bulk insert
            execute_values(
                cursor,
                """
                INSERT INTO customer_data (
                    customer_id, unnamed_0, x, customer, traintest, churndep, revenue, mou, recchrge, directas, overage, roam,
                    changem, changer, dropvce, blckvce, unansvce, custcare, threeway,
                    mourec, outcalls, incalls, peakvce, opeakvce, dropblk, callfwdv,
                    callwait, months, uniqsubs, actvsubs, phones, models, eqpdays,
                    age1, age2, refer, income, setprc,
                    children, credita, creditaa, prizmrur, prizmub, prizmtwn, refurb,
                    webcap, truck, rv, occprof, occcler, occcrft, occstud, occhmkr,
                    occret, occself, ownrent, marryun, marryyes, mailord, mailres,
                    mailflag, travel, pcown, creditcd, newcelly, newcelln, incmiss,
                    mcycle, setprcm, retcall, retcalls, retaccpt,
                    churn, source, timestamp, batch_id
                ) VALUES %s
                ON CONFLICT (customer_id, timestamp) DO NOTHING
                """,
                values
            )
            
            saved_count = cursor.rowcount
            cursor.close()
            
            return saved_count, len(customers) - saved_count
        
    except Exception as e:
        logger.error(f"Error saving batch customer data: {str(e)}")
        return 0, len(customers)
