-- Source fields are TEXT to preserve raw values for the quality audit.
SELECT ticket_id, customer_id, created_at, available_at,
       topic, channel, resolved_at, resolution_available_at,
       satisfaction_score
FROM support_current_raw;

-- Diagnostic queries; they do not replace full duplicate analysis.
SELECT COUNT(*) AS source_rows FROM support_current_raw;

SELECT ticket_id, COUNT(*) AS copies
FROM support_current_raw
GROUP BY ticket_id
HAVING COUNT(*) > 1;

SELECT COUNT(*) AS empty_customer_id
FROM support_current_raw
WHERE customer_id IS NULL OR TRIM(customer_id) = '';

SELECT topic, COUNT(*) AS rows_by_topic
FROM support_current_raw
GROUP BY topic;
