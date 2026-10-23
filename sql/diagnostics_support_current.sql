-- Диагностические запросы к support_current_raw (запускаются из src/load.py,
-- результаты сохраняются в reports/stage2/sqlite_diagnostics.json).

-- source_rows
SELECT COUNT(*) AS source_rows FROM support_current_raw;

-- duplicated_ticket_ids
SELECT COUNT(*) AS duplicated_ticket_ids FROM (
    SELECT ticket_id FROM support_current_raw GROUP BY ticket_id HAVING COUNT(*) > 1
);

-- full_duplicate_copies
SELECT COUNT(*) - COUNT(DISTINCT ticket_id || '|' || IFNULL(customer_id, '') || '|' || created_at || '|' ||
       available_at || '|' || IFNULL(topic, '') || '|' || channel || '|' || IFNULL(resolved_at, '') || '|' ||
       IFNULL(resolution_available_at, '') || '|' || IFNULL(satisfaction_score, '')) AS full_duplicate_copies
FROM support_current_raw;

-- empty_customer_id
SELECT COUNT(*) AS empty_customer_id FROM support_current_raw
WHERE customer_id IS NULL OR TRIM(customer_id) = '';

-- created_without_timezone
SELECT COUNT(*) AS created_without_timezone FROM support_current_raw
WHERE created_at NOT LIKE '%Z';
