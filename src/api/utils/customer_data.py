from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class TrainTestSplit(str, Enum):
    """Train/test split indicator"""
    TRAIN = 0
    TEST = 1
class CustomerData(BaseModel):
    """Customer data model for Cell2Cell churn prediction"""
    
    # COLUMNS TO DROP DURING PREPROCESSING
    unnamed_0: Optional[int] = Field(None, description="Unnamed index column", alias="Unnamed: 0")
    x: Optional[int] = Field(None, description="X column (likely duplicate index)")
    customer: Optional[str] = Field(None, description="Customer identifier")
    traintest: Optional[TrainTestSplit] = Field(None, description="Train/test split indicator")
    churndep: Optional[str] = Field(None, description="Dependent churn variable (alternative target)")
    # DEMOGRAPHIC INFORMATION 
    age1: Optional[float] = Field(None, description="Age of primary account holder")
    age2: Optional[float] = Field(None, description="Age of secondary account holder")
    children: Optional[str] = Field(None, description="Children in household")
    income: Optional[float] = Field(None, description="Household income")
    ownrent: Optional[str] = Field(None, description="Home ownership status")
    
    # CREDIT INFORMATION 
    credita: Optional[str] = Field(None, description="Credit rating A")
    creditaa: Optional[str] = Field(None, description="Credit rating AA")
    creditcd: Optional[str] = Field(None, description="Credit card indicator")
    incmiss: Optional[str] = Field(None, description="Missing income indicator")
    
    # GEOGRAPHIC/LIFESTYLE SEGMENTS
    prizmrur: Optional[str] = Field(None, description="Prizm rural segment")
    prizmub: Optional[str] = Field(None, description="Prizm urban segment")
    prizmtwn: Optional[str] = Field(None, description="Prizm town segment")
    
    # OCCUPATION
    occprof: Optional[str] = Field(None, description="Professional occupation")
    occcler: Optional[str] = Field(None, description="Clerical occupation")
    occcrft: Optional[str] = Field(None, description="Craft occupation")
    occstud: Optional[str] = Field(None, description="Student occupation")
    occhmkr: Optional[str] = Field(None, description="Homemaker occupation")
    occret: Optional[str] = Field(None, description="Retired occupation")
    occself: Optional[str] = Field(None, description="Self-employed occupation")
    
    # MARITAL STATUS
    marryun: Optional[str] = Field(None, description="Unmarried indicator")
    marryyes: Optional[str] = Field(None, description="Married indicator")
    
    # VEHICLE OWNERSHIP
    truck: Optional[str] = Field(None, description="Truck owner")
    rv: Optional[str] = Field(None, description="RV owner")
    mcycle: Optional[str] = Field(None, description="Motorcycle owner")
    
    # MAILING PREFERENCES
    mailord: Optional[str] = Field(None, description="Mail order buyer")
    mailres: Optional[str] = Field(None, description="Mail respondent")
    mailflag: Optional[str] = Field(None, description="Mail flag")
    
    # TECHNOLOGY USAGE
    travel: Optional[str] = Field(None, description="Travel user")
    pcown: Optional[str] = Field(None, description="PC owner")
    webcap: Optional[str] = Field(None, description="Web capability")
    
    # EQUIPMENT INFORMATION
    refurb: Optional[str] = Field(None, description="Refurbished phone")
    phones: Optional[int] = Field(None, description="Number of phones")
    models: Optional[int] = Field(None, description="Number of models")
    eqpdays: Optional[int] = Field(None, description="Days since equipment change")
    
    # SERVICE USAGE METRICS
    revenue: Optional[float] = Field(None, description="Monthly revenue")
    mou: Optional[float] = Field(None, description="Minutes of use")
    recchrge: Optional[float] = Field(None, description="Recurring charge")
    directas: Optional[float] = Field(None, description="Direct assistant usage")
    overage: Optional[float] = Field(None, description="Overage minutes")
    roam: Optional[float] = Field(None, description="Roaming charges")
    
    # CALL PATTERN METRICS
    changem: Optional[float] = Field(None, description="Change in minutes")
    changer: Optional[float] = Field(None, description="Change in revenue")
    dropvce: Optional[float] = Field(None, description="Dropped voice calls")
    blckvce: Optional[float] = Field(None, description="Blocked voice calls")
    unansvce: Optional[float] = Field(None, description="Unanswered voice calls")
    custcare: Optional[float] = Field(None, description="Customer care calls")
    threeway: Optional[float] = Field(None, description="Three-way calls")
    mourec: Optional[float] = Field(None, description="Received minutes of use")
    outcalls: Optional[float] = Field(None, description="Outgoing calls")
    incalls: Optional[float] = Field(None, description="Incoming calls")
    peakvce: Optional[float] = Field(None, description="Peak voice calls")
    opeakvce: Optional[float] = Field(None, description="Off-peak voice calls")
    dropblk: Optional[float] = Field(None, description="Dropped or blocked calls")
    callfwdv: Optional[float] = Field(None, description="Call forwarding voice")
    callwait: Optional[float] = Field(None, description="Call waiting usage")
    
    # ACCOUNT INFORMATION
    months: Optional[int] = Field(None, description="Months with service")
    uniqsubs: Optional[int] = Field(None, description="Unique subscribers")
    actvsubs: Optional[int] = Field(None, description="Active subscribers")
    
    # REFERRAL AND RETENTION
    refer: Optional[int] = Field(None, description="Referrals made")
    retcall: Optional[str] = Field(None, description="Retention call indicator")
    retcalls: Optional[str] = Field(None, description="Retention calls")
    retaccpt: Optional[str] = Field(None, description="Retention offer accepted")
    
    # PRICING AND PLANS
    setprc: Optional[float] = Field(None, description="Set price")
    setprcm: Optional[str] = Field(None, description="Set price modified")
    
    # NEW EQUIPMENT
    newcelly: Optional[str] = Field(None, description="New cell phone yes")
    newcelln: Optional[str] = Field(None, description="New cell phone no")
    
    # TARGET VARIABLE
    churn: Optional[str] = Field(None, description="Churn indicator")
    
    # Metadata
    source: Optional[str] = Field(default="api", description="Data source: api, csv, excel, google_form")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @field_validator(
        "children", "credita", "creditaa", "prizmrur", "prizmub", "prizmtwn",
        "refurb", "webcap", "truck", "rv", "occprof", "occcler", "occcrft",
        "occstud", "occhmkr", "occret", "occself", "ownrent", "marryun",
        "marryyes", "mailord", "mailres", "mailflag", "travel", "pcown",
        "creditcd", "newcelly", "newcelln", "incmiss", "mcycle", "setprcm",
        "retcall", "retcalls", "retaccpt", "churn", "churndep", "customer"
    )
    def convert_to_string(cls, v):
        """Convert numeric values to strings for categorical fields"""
        if v is not None:
            return str(v)
        return v
    
    # Non-negative numerical fields

    @field_validator(
        "revenue", "mou", "recchrge", "directas", "overage", "roam",
        "dropvce", "blckvce", "unansvce", "custcare", "threeway",
        "mourec", "outcalls", "incalls", "peakvce", "opeakvce",
        "dropblk", "callfwdv", "callwait", "months", "uniqsubs",
        "actvsubs", "phones", "models", "eqpdays", "refer", "setprc"
    )

    def non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"{v} must be >= 0")
        return v

    # Age constraints
    @field_validator("age1", "age2")
    def valid_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError("Age must be between 0 and 120")
        return v

    @model_validator(mode="after")
    def age_order(cls, self):
        age1 = self.age1
        age2 = self.age2
        if age1 is not None and age2 is not None and age1 < age2:
            raise ValueError("age1 must be >= age2")
        return self

    # Income: 0–9 scale
    @field_validator("income")
    def valid_income(cls, v):
        if v is not None and (v < 0 or v > 9):
            raise ValueError("income must be between 1 and 9")
        return v

    # Binary categorical fields must be 0 or 1
    _binary_fields = [
        "children", "credita", "creditaa", "prizmrur", "prizmub", "prizmtwn",
        "refurb", "webcap", "truck", "rv", "occprof", "occcler", "occcrft",
        "occstud", "occhmkr", "occret", "occself", "ownrent", "marryun",
        "marryyes", "mailord", "mailres", "mailflag", "travel", "pcown",
        "creditcd", "newcelly", "newcelln", "incmiss", "mcycle", "setprcm",
        "retcall", "retaccpt"
    ]

    @field_validator(*_binary_fields)
    def is_binary(cls, v):
        if v is not None and v not in ("0", "1"):
            raise ValueError(f"{v} must be 0 or 1")
        return v

    # 5. Mutually exclusive: newcelly and newcelln
    @model_validator(mode="after")
    def newcell_exclusive(cls, self):
        y = self.newcelly
        n = self.newcelln
        if y == 1 and n == 1:
            raise ValueError("newcelly and newcelln cannot both be 1")
        if y is None and n is None:
            raise ValueError("One of newcelly or newcelln must be set")
        return self

    # Subscriber logic
    @model_validator(mode="after")
    def subscriber_logic(cls, self):
        uniq = self.uniqsubs
        actv = self.actvsubs
        if uniq is not None and actv is not None:
            if actv > uniq:
                raise ValueError("actvsubs cannot exceed uniqsubs")
            if actv < 1:
                raise ValueError("actvsubs must be at least 1")
        return self

    # Retention call logic
    @model_validator(mode="after")
    def retention_logic(cls, self):
        retcall = self.retcall
        retcalls = self.retcalls
        if retcall == 1 and (retcalls is None or retcalls < 1):
            raise ValueError("If retcall=1, retcalls must be >= 1")
        return self

    @field_validator("churn")
    def valid_churn(cls, v):
        if v is not None:
            # Handle both string "0"/"1" and integer 0/1
            if isinstance(v, str):
                if v not in ("0", "1"):
                    raise ValueError("churn must be '0' or '1'")
                return v
            elif isinstance(v, int):
                if v not in (0, 1):
                    raise ValueError("churn must be 0 or 1")
                return str(v)  # Convert to string for consistency
            else:
                raise ValueError("churn must be 0 or 1")
        return v

    class Config:
        json_schema_extra = {
            "example_no_churn": {'unnamed_0': 50001.0,
                        'x': 50001.0,
                        'customer': 1095590.0,
                        'traintest': 1.0,
                        'churn': 0.0,
                        'churndep': 0.0,
                        'revenue': 42.58250046,
                        'mou': 387.0,
                        'recchrge': 44.99000168,
                        'directas': 0.495000005,
                        'overage': 17.75,
                        'roam': 0.0,
                        'changem': -95.0,
                        'changer': -7.59250021,
                        'dropvce': 10.33333302,
                        'blckvce': 0.666666687,
                        'unansvce': 31.33333397,
                        'custcare': 0.666666687,
                        'threeway': 0.0,
                        'mourec': 124.3799973,
                        'outcalls': 43.33333206,
                        'incalls': 0.0,
                        'peakvce': 94.33333588,
                        'opeakvce': 130.0,
                        'dropblk': 11.0,
                        'callfwdv': 0.0,
                        'callwait': 0.0,
                        'months': 9.0,
                        'uniqsubs': 1.0,
                        'actvsubs': 1.0,
                        'phones': 2.0,
                        'models': 1.0,
                        'eqpdays': 108.0,
                        'age1': 0.0,
                        'age2': 0.0,
                        'children': 0.0,
                        'credita': 1.0,
                        'creditaa': 0.0,
                        'prizmrur': 0.0,
                        'prizmub': 1.0,
                        'prizmtwn': 0.0,
                        'refurb': 0.0,
                        'webcap': 1.0,
                        'truck': 0.0,
                        'rv': 0.0,
                        'occprof': 0.0,
                        'occcler': 0.0,
                        'occcrft': 0.0,
                        'occstud': 0.0,
                        'occhmkr': 0.0,
                        'occret': 0.0,
                        'occself': 0.0,
                        'ownrent': 0.0,
                        'marryun': 0.0,
                        'marryyes': 0.0,
                        'mailord': 0.0,
                        'mailres': 0.0,
                        'mailflag': 0.0,
                        'travel': 0.0,
                        'pcown': 0.0,
                        'creditcd': 1.0,
                        'retcalls': 0.0,
                        'retaccpt': 0.0,
                        'newcelly': 0.0,
                        'newcelln': 0.0,
                        'refer': 0.0,
                        'incmiss': 0.0,
                        'income': 2.0,
                        'mcycle': 0.0,
                        'setprcm': 0.0,
                        'setprc': 129.9899902,
                        'retcall': 0.0},
            "example_churn": {'unnamed_0': 60001.0,
                                'x': 60001.0,
                                'customer': 1043968.0,
                                'traintest': 1.0,
                                'churn': 1.0,
                                'churndep': 1.0,
                                'revenue': 62.29000092,
                                'mou': 861.0,
                                'recchrge': 44.99000168,
                                'directas': 0.0,
                                'overage': 78.0,
                                'roam': 0.0,
                                'changem': -638.0,
                                'changer': -27.29999924,
                                'dropvce': 16.33333397,
                                'blckvce': 9.666666985,
                                'unansvce': 34.0,
                                'custcare': 1.666666627,
                                'threeway': 0.0,
                                'mourec': 438.4933472,
                                'outcalls': 76.33333588,
                                'incalls': 26.33333397,
                                'peakvce': 99.33333588,
                                'opeakvce': 193.3333282,
                                'dropblk': 26.0,
                                'callfwdv': 0.0,
                                'callwait': 5.0,
                                'months': 19.0,
                                'uniqsubs': 1.0,
                                'actvsubs': 1.0,
                                'phones': 1.0,
                                'models': 1.0,
                                'eqpdays': 556.0,
                                'age1': 40.0,
                                'age2': 42.0,
                                'children': 1.0,
                                'credita': 0.0,
                                'creditaa': 0.0,
                                'prizmrur': 0.0,
                                'prizmub': 0.0,
                                'prizmtwn': 1.0,
                                'refurb': 0.0,
                                'webcap': 1.0,
                                'truck': 1.0,
                                'rv': 0.0,
                                'occprof': 0.0,
                                'occcler': 0.0,
                                'occcrft': 0.0,
                                'occstud': 0.0,
                                'occhmkr': 0.0,
                                'occret': 0.0,
                                'occself': 0.0,
                                'ownrent': 0.0,
                                'marryun': 0.0,
                                'marryyes': 1.0,
                                'mailord': 0.0,
                                'mailres': 0.0,
                                'mailflag': 0.0,
                                'travel': 0.0,
                                'pcown': 1.0,
                                'creditcd': 1.0,
                                'retcalls': 0.0,
                                'retaccpt': 0.0,
                                'newcelly': 0.0,
                                'newcelln': 0.0,
                                'refer': 0.0,
                                'incmiss': 0.0,
                                'income': 3.0,
                                'mcycle': 0.0,
                                'setprcm': 1.0,
                                'setprc': 0.0,
                                'retcall': 0.0}
                                }
class BatchCustomerData(BaseModel):
    """Batch customer data entries"""
    customers: List[CustomerData]
    
    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "churn": 1,
                        "revenue": 45.2,
                        "mou": 320.1,
                        "months": 6,
                        "dropvce": 8.3,
                        "custcare": 4.2,
                        "creditaa": 0,
                        "newcelly": 1
                    }
                ]
            }
        }

