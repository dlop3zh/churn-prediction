-- =============================================================================
-- Setup en Snowflake: stage de artefactos y tabla de scoring.
-- Sustituye <DB> y <SCHEMA> por los de tu entorno.
-- (La tabla también se crea automáticamente en la primera inferencia.)
-- =============================================================================

CREATE STAGE IF NOT EXISTS <DB>.<SCHEMA>.CHURN_ARTIFACTS
    COMMENT = 'Artefactos del modelo de churn (bundle joblib) y código del SP';

CREATE TABLE IF NOT EXISTS <DB>.<SCHEMA>.CHURN_SCORING_MENSUAL (
    CLIENTE_ID                VARCHAR,
    PAIS                      VARCHAR,
    IDCENTRO_PRINCIPAL        VARCHAR,
    CENTRO_PRINCIPAL          VARCHAR,
    N_CENTROS_DISTINTOS       INTEGER,
    MONETARY                  VARCHAR,
    ANIO_MES                  INTEGER,
    FECHA_ULTIMA_VISITA       DATE,
    PROB_CHURN_HARD           FLOAT,
    PRED_CHURN_HARD           INTEGER,
    SEGMENTO_RIESGO           VARCHAR,
    MODELO                    VARCHAR,     -- 'general' | 'vip'
    CLIENTE_NUEVO             INTEGER,
    N_SESIONES_MES            INTEGER,
    RTP_MEDIO                 FLOAT,
    COIN_IN_TOTAL             FLOAT,
    RATIO_ADVANTAGE           FLOAT,
    RACHA_MAX_SESIONES_DURAS  INTEGER,
    DIAS_ENTRE_SESIONES_MEDIO FLOAT,
    FECHA_SCORING             INTEGER
);
