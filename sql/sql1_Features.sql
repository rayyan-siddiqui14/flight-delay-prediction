CREATE OR REPLACE TABLE features AS
WITH legs AS (
    SELECT *,
        LAG(ArrDelay)   OVER w AS prev_arr_delay,
        LAG(CRSArrTime) OVER w AS prev_sched_arr,
        ROW_NUMBER()    OVER w AS leg_number
    FROM flights
    WINDOW w AS (PARTITION BY tail, flight_date ORDER BY CRSDepTime)
)
SELECT
    l.flight_date, l.carrier, l.Origin, l.Dest, l.tail,

    -- schedule features
    l.CRSDepTime // 100                            AS dep_hour,
    l.CRSDepTime %  100                            AS dep_minute,
    CAST(strftime(l.flight_date, '%w') AS INTEGER) AS day_of_week,
    CAST(strftime(l.flight_date, '%m') AS INTEGER) AS month,
    l.Distance,
    l.CRSElapsedTime,

    -- propagation features
    COALESCE(l.prev_arr_delay, 0)                  AS prev_arr_delay,
    CASE WHEN l.prev_arr_delay IS NULL THEN 1 ELSE 0 END AS is_first_leg,
    l.leg_number,
    CASE WHEN l.prev_sched_arr IS NULL THEN NULL
         ELSE (l.CRSDepTime // 100 * 60 + l.CRSDepTime % 100)
            - (l.prev_sched_arr // 100 * 60 + l.prev_sched_arr % 100)
    END                                            AS turnaround_min,

    -- historical features
    COALESCE(r.route_delay_rate,     0.22)         AS route_delay_rate,
    COALESCE(r.route_mean_delay,     8.0)          AS route_mean_delay,
    COALESCE(c.carrier_delay_rate,   0.22)         AS carrier_delay_rate,
    COALESCE(ao.origin_delay_rate,   0.22)         AS origin_delay_rate,
    COALESCE(ao.avg_daily_departures, 50)          AS origin_daily_departures,
    COALESCE(ad.origin_delay_rate,   0.22)         AS dest_delay_rate,
    COALESCE(ah.departures_this_hour, 5)           AS departures_this_hour,
    COALESCE(ah.hour_delay_rate,     0.22)         AS hour_delay_rate,

    -- target
    l.ArrDel15                                     AS target,
    l.ArrDelay                                     AS arr_delay_min
FROM legs l
LEFT JOIN route_hist        r  ON l.Origin = r.Origin AND l.Dest = r.Dest
LEFT JOIN carrier_hist      c  ON l.carrier = c.carrier
LEFT JOIN airport_hist      ao ON l.Origin = ao.airport
LEFT JOIN airport_hist      ad ON l.Dest = ad.airport
LEFT JOIN airport_hour_hist ah ON l.Origin = ah.airport AND l.CRSDepTime // 100 = ah.hour
WHERE l.ArrDel15 IS NOT NULL;