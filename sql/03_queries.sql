-- =============================================================================
-- Consultas de explotación sobre la tabla de scoring.
-- =============================================================================

-- Clientes en riesgo alto del último mes puntuado, ordenados por probabilidad
SELECT CLIENTE_ID, MONETARY, CENTRO_PRINCIPAL, PROB_CHURN_HARD, MODELO,
       N_SESIONES_MES, RTP_MEDIO, FECHA_ULTIMA_VISITA
FROM   <DB>.<SCHEMA>.CHURN_SCORING_MENSUAL
WHERE  ANIO_MES = (SELECT MAX(ANIO_MES) FROM <DB>.<SCHEMA>.CHURN_SCORING_MENSUAL)
  AND  SEGMENTO_RIESGO = 'Alto'
ORDER  BY PROB_CHURN_HARD DESC;

-- Distribución de riesgo por mes y modelo
SELECT ANIO_MES, MODELO, SEGMENTO_RIESGO, COUNT(*) AS N_CLIENTES
FROM   <DB>.<SCHEMA>.CHURN_SCORING_MENSUAL
GROUP  BY 1, 2, 3
ORDER  BY 1 DESC, 2, 3;

-- Seguimiento: ¿volvieron los clientes marcados el mes M?
-- (precisión real en producción, comparando con la actividad del mes M+1)
WITH marcados AS (
    SELECT CLIENTE_ID, ANIO_MES, MODELO, FECHA_ULTIMA_VISITA
    FROM   <DB>.<SCHEMA>.CHURN_SCORING_MENSUAL
    WHERE  PRED_CHURN_HARD = 1
),
proxima AS (
    SELECT m.CLIENTE_ID, m.ANIO_MES, m.MODELO,
           MIN(s.SESSION_DATE) AS PROXIMA_VISITA,
           ANY_VALUE(m.FECHA_ULTIMA_VISITA) AS ULTIMA_VISITA
    FROM   marcados m
    LEFT JOIN <DB>.<SCHEMA>.<SESSIONS_VIEW> s
           ON s.CUSTOMER_ID = m.CLIENTE_ID
          AND s.SESSION_DATE > m.FECHA_ULTIMA_VISITA
    GROUP  BY 1, 2, 3
)
SELECT ANIO_MES, MODELO,
       COUNT(*) AS MARCADOS,
       AVG(IFF(PROXIMA_VISITA IS NULL
               OR DATEDIFF('day', ULTIMA_VISITA, PROXIMA_VISITA) > 30, 1, 0)) AS PRECISION_REAL
FROM   proxima
GROUP  BY 1, 2
ORDER  BY 1 DESC, 2;
