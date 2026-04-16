-- telecom customer intelligence platform
-- star schema DDL

-- ------------------------------------------------------------------ --
-- dimension tables
-- ------------------------------------------------------------------ --

CREATE TABLE dim_customer (
    customer_sk     SERIAL PRIMARY KEY,
    customer_id     VARCHAR(10) NOT NULL UNIQUE,
    age             SMALLINT,
    gender          CHAR(1),
    city            VARCHAR(50),
    contract_type   VARCHAR(20),
    payment_method  VARCHAR(30),
    paperless       BOOLEAN,
    has_internet    BOOLEAN,
    has_tv          BOOLEAN,
    has_roaming     BOOLEAN,
    segment         SMALLINT,
    segment_name    VARCHAR(30),
    churn_flag      BOOLEAN,
    valid_from      DATE NOT NULL DEFAULT CURRENT_DATE,
    valid_to        DATE
);

CREATE TABLE dim_tariff (
    tariff_sk       SERIAL PRIMARY KEY,
    tariff_id       VARCHAR(5) NOT NULL UNIQUE,
    tariff_name     VARCHAR(50),
    category        VARCHAR(20),
    monthly_price   NUMERIC(10, 2),
    data_gb         SMALLINT,
    minutes         SMALLINT
);

CREATE TABLE dim_date (
    date_sk         INT PRIMARY KEY,  -- YYYYMMDD integer key
    full_date       DATE NOT NULL UNIQUE,
    year            SMALLINT,
    quarter         SMALLINT,
    month           SMALLINT,
    month_name      VARCHAR(10),
    week_of_year    SMALLINT,
    day_of_month    SMALLINT,
    day_of_week     SMALLINT,
    is_weekend      BOOLEAN,
    is_month_end    BOOLEAN
);

CREATE TABLE dim_city (
    city_sk     SERIAL PRIMARY KEY,
    city_name   VARCHAR(50) NOT NULL UNIQUE,
    region      VARCHAR(50),
    country     VARCHAR(30) DEFAULT 'Kazakhstan'
);

-- ------------------------------------------------------------------ --
-- fact table
-- ------------------------------------------------------------------ --

CREATE TABLE fact_transactions (
    transaction_sk  BIGSERIAL PRIMARY KEY,
    transaction_id  VARCHAR(15) NOT NULL UNIQUE,
    customer_sk     INT NOT NULL REFERENCES dim_customer(customer_sk),
    tariff_sk       INT NOT NULL REFERENCES dim_tariff(tariff_sk),
    date_sk         INT NOT NULL REFERENCES dim_date(date_sk),
    city_sk         INT REFERENCES dim_city(city_sk),
    amount          NUMERIC(12, 2) NOT NULL,
    data_gb_used    NUMERIC(8, 2),
    minutes_used    NUMERIC(10, 1)
);

-- ------------------------------------------------------------------ --
-- indexes
-- ------------------------------------------------------------------ --

CREATE INDEX idx_fact_customer ON fact_transactions(customer_sk);
CREATE INDEX idx_fact_date     ON fact_transactions(date_sk);
CREATE INDEX idx_fact_tariff   ON fact_transactions(tariff_sk);
CREATE INDEX idx_dim_customer_id ON dim_customer(customer_id);

-- ------------------------------------------------------------------ --
-- populate dim_date for 2024-2026
-- ------------------------------------------------------------------ --

INSERT INTO dim_date (
    date_sk, full_date, year, quarter, month, month_name,
    week_of_year, day_of_month, day_of_week, is_weekend, is_month_end
)
SELECT
    TO_CHAR(d, 'YYYYMMDD')::INT,
    d,
    EXTRACT(YEAR FROM d)::SMALLINT,
    EXTRACT(QUARTER FROM d)::SMALLINT,
    EXTRACT(MONTH FROM d)::SMALLINT,
    TO_CHAR(d, 'Month'),
    EXTRACT(WEEK FROM d)::SMALLINT,
    EXTRACT(DAY FROM d)::SMALLINT,
    EXTRACT(DOW FROM d)::SMALLINT,
    EXTRACT(DOW FROM d) IN (0, 6),
    d = DATE_TRUNC('month', d) + INTERVAL '1 month' - INTERVAL '1 day'
FROM GENERATE_SERIES('2024-01-01'::DATE, '2026-12-31'::DATE, '1 day') d;
