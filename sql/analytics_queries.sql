-- telecom analytics queries
-- demonstrates CTEs, window functions, cohort analysis

-- ------------------------------------------------------------------ --
-- 1. monthly revenue with MoM growth (LAG)
-- ------------------------------------------------------------------ --

WITH monthly_rev AS (
    SELECT
        d.year,
        d.month,
        d.month_name,
        SUM(f.amount)                           AS revenue,
        COUNT(DISTINCT f.customer_sk)           AS active_customers
    FROM fact_transactions f
    JOIN dim_date d ON f.date_sk = d.date_sk
    GROUP BY d.year, d.month, d.month_name
),
with_growth AS (
    SELECT
        *,
        LAG(revenue) OVER (ORDER BY year, month)        AS prev_revenue,
        LAG(active_customers) OVER (ORDER BY year, month) AS prev_customers
    FROM monthly_rev
)
SELECT
    year,
    month,
    month_name,
    ROUND(revenue, 0)                               AS revenue,
    active_customers,
    ROUND(
        100.0 * (revenue - prev_revenue) / NULLIF(prev_revenue, 0), 2
    )                                               AS mom_growth_pct,
    ROUND(revenue / NULLIF(active_customers, 0), 2) AS arpu
FROM with_growth
ORDER BY year, month;


-- ------------------------------------------------------------------ --
-- 2. customer value tiers with NTILE + running total
-- ------------------------------------------------------------------ --

WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        c.contract_type,
        c.segment_name,
        SUM(f.amount)   AS total_spend,
        COUNT(f.*)      AS months_active
    FROM fact_transactions f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    GROUP BY c.customer_id, c.city, c.contract_type, c.segment_name
)
SELECT
    customer_id,
    city,
    contract_type,
    segment_name,
    ROUND(total_spend, 0)                                   AS total_spend,
    months_active,
    NTILE(4) OVER (ORDER BY total_spend DESC)               AS spend_quartile,
    ROUND(
        100.0 * SUM(total_spend) OVER (
            ORDER BY total_spend DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / SUM(total_spend) OVER (), 2
    )                                                       AS cumulative_pct
FROM customer_spend
ORDER BY total_spend DESC;


-- ------------------------------------------------------------------ --
-- 3. top customer per city (ROW_NUMBER)
-- ------------------------------------------------------------------ --

WITH city_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(f.amount)   AS total_spend
    FROM fact_transactions f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    GROUP BY c.customer_id, c.city
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rn
    FROM city_spend
)
SELECT city, customer_id, ROUND(total_spend, 0) AS total_spend
FROM ranked
WHERE rn <= 3
ORDER BY city, rn;


-- ------------------------------------------------------------------ --
-- 4. tariff upgrade/downgrade detection (LAG on tariff price)
-- ------------------------------------------------------------------ --

WITH tariff_history AS (
    SELECT
        c.customer_id,
        d.full_date,
        t.tariff_id,
        t.monthly_price,
        LAG(t.monthly_price) OVER (
            PARTITION BY c.customer_id ORDER BY d.full_date
        )                                               AS prev_price,
        LAG(t.tariff_id) OVER (
            PARTITION BY c.customer_id ORDER BY d.full_date
        )                                               AS prev_tariff
    FROM fact_transactions f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    JOIN dim_tariff   t ON f.tariff_sk   = t.tariff_sk
    JOIN dim_date     d ON f.date_sk     = d.date_sk
)
SELECT
    customer_id,
    full_date                                           AS change_date,
    prev_tariff                                         AS from_tariff,
    tariff_id                                           AS to_tariff,
    ROUND(prev_price, 0)                                AS from_price,
    ROUND(monthly_price, 0)                             AS to_price,
    CASE
        WHEN monthly_price > prev_price THEN 'upgrade'
        WHEN monthly_price < prev_price THEN 'downgrade'
        ELSE 'lateral'
    END                                                 AS change_type
FROM tariff_history
WHERE prev_tariff IS NOT NULL
  AND prev_tariff <> tariff_id
ORDER BY customer_id, change_date;


-- ------------------------------------------------------------------ --
-- 5. cohort retention (first transaction month as cohort)
-- ------------------------------------------------------------------ --

WITH first_month AS (
    SELECT
        c.customer_id,
        MIN(d.year * 100 + d.month)     AS cohort_ym,
        MIN(d.full_date)                AS first_date
    FROM fact_transactions f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    JOIN dim_date     d ON f.date_sk     = d.date_sk
    GROUP BY c.customer_id
),
activity AS (
    SELECT
        fm.customer_id,
        fm.cohort_ym,
        d.year * 100 + d.month                          AS activity_ym,
        -- months since cohort start (approximate)
        (d.year * 12 + d.month) - 
        (fm.cohort_ym / 100 * 12 + fm.cohort_ym % 100) AS period_num
    FROM fact_transactions f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    JOIN dim_date     d ON f.date_sk     = d.date_sk
    JOIN first_month fm ON c.customer_id = fm.customer_id
),
cohort_size AS (
    SELECT cohort_ym, COUNT(DISTINCT customer_id) AS n_customers
    FROM first_month
    GROUP BY cohort_ym
)
SELECT
    a.cohort_ym,
    cs.n_customers                                          AS cohort_size,
    a.period_num,
    COUNT(DISTINCT a.customer_id)                           AS retained,
    ROUND(
        100.0 * COUNT(DISTINCT a.customer_id) / cs.n_customers, 1
    )                                                       AS retention_pct
FROM activity a
JOIN cohort_size cs ON a.cohort_ym = cs.cohort_ym
GROUP BY a.cohort_ym, cs.n_customers, a.period_num
ORDER BY a.cohort_ym, a.period_num;


-- ------------------------------------------------------------------ --
-- 6. churn risk score by segment (RANK + CASE)
-- ------------------------------------------------------------------ --

WITH segment_stats AS (
    SELECT
        c.segment_name,
        COUNT(DISTINCT c.customer_id)                       AS n_customers,
        SUM(c.churn_flag::INT)                              AS churned,
        ROUND(AVG(f.amount), 2)                             AS avg_monthly_spend,
        ROUND(AVG(
            EXTRACT(MONTH FROM AGE(CURRENT_DATE, fc.first_date))
        ), 1)                                               AS avg_tenure_months
    FROM dim_customer c
    JOIN fact_transactions f ON c.customer_sk = f.customer_sk
    JOIN (
        SELECT customer_sk, MIN(d.full_date) AS first_date
        FROM fact_transactions ft
        JOIN dim_date d ON ft.date_sk = d.date_sk
        GROUP BY customer_sk
    ) fc ON c.customer_sk = fc.customer_sk
    GROUP BY c.segment_name
)
SELECT
    segment_name,
    n_customers,
    churned,
    ROUND(100.0 * churned / NULLIF(n_customers, 0), 1)     AS churn_rate_pct,
    avg_monthly_spend,
    avg_tenure_months,
    RANK() OVER (ORDER BY churned::FLOAT / NULLIF(n_customers, 0) DESC) AS churn_risk_rank,
    CASE
        WHEN churned::FLOAT / NULLIF(n_customers, 0) > 0.25 THEN 'HIGH'
        WHEN churned::FLOAT / NULLIF(n_customers, 0) > 0.15 THEN 'MEDIUM'
        ELSE 'LOW'
    END                                                     AS risk_level
FROM segment_stats
ORDER BY churn_risk_rank;


-- ------------------------------------------------------------------ --
-- 7. LEAD: days until next transaction per customer
-- ------------------------------------------------------------------ --

WITH ordered_tx AS (
    SELECT
        c.customer_id,
        d.full_date                                         AS tx_date,
        f.amount,
        LEAD(d.full_date) OVER (
            PARTITION BY c.customer_id ORDER BY d.full_date
        )                                                   AS next_tx_date
    FROM fact_transactions f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    JOIN dim_date     d ON f.date_sk     = d.date_sk
)
SELECT
    customer_id,
    tx_date,
    ROUND(amount, 0)                                        AS amount,
    next_tx_date,
    (next_tx_date - tx_date)                                AS days_to_next
FROM ordered_tx
ORDER BY customer_id, tx_date;
