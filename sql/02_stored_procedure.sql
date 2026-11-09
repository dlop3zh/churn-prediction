-- =============================================================================
-- Stored Procedure de inferencia mensual + Task programada.
--
-- Antes: sube el paquete y la config al stage con
--     python scripts/deploy_snowflake.py --config config/config.yaml
--
-- IMPORTANTE: la versión de scikit-learn / xgboost del SP debe coincidir con la
-- usada al entrenar (el bundle guarda `sklearn_version` para comprobarlo).
-- =============================================================================

CREATE OR REPLACE PROCEDURE <DB>.<SCHEMA>.SP_CHURN_INFERENCIA(TARGET_MONTH INTEGER DEFAULT 0)
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.11'
    PACKAGES = ('snowflake-snowpark-python', 'pandas', 'numpy', 'scikit-learn',
                'xgboost', 'joblib', 'pyyaml')
    IMPORTS = ('@<DB>.<SCHEMA>.CHURN_ARTIFACTS/churn.zip',
               '@<DB>.<SCHEMA>.CHURN_ARTIFACTS/config.yaml')
    HANDLER = 'churn.inference.sp_handler'
    EXECUTE AS CALLER;

-- Ejecución manual (0 = último mes cerrado)
CALL <DB>.<SCHEMA>.SP_CHURN_INFERENCIA();
CALL <DB>.<SCHEMA>.SP_CHURN_INFERENCIA(202610);

-- Ejecución automática: día 3 de cada mes a las 07:00
CREATE OR REPLACE TASK <DB>.<SCHEMA>.TASK_CHURN_INFERENCIA
    WAREHOUSE = <WAREHOUSE>
    SCHEDULE  = 'USING CRON 0 7 3 * * Europe/Madrid'
AS
    CALL <DB>.<SCHEMA>.SP_CHURN_INFERENCIA();

ALTER TASK <DB>.<SCHEMA>.TASK_CHURN_INFERENCIA RESUME;
