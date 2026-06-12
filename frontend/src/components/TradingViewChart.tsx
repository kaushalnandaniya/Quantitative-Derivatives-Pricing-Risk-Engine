"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, CrosshairMode, ISeriesApi, CandlestickSeries, createSeriesMarkers } from "lightweight-charts";

export interface OHLCV {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TradeMarker {
  time: string;
  position: "aboveBar" | "belowBar" | "inBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle" | "square";
  text: string;
}

interface TradingViewChartProps {
  data: OHLCV[];
  markers?: TradeMarker[];
}

export default function TradingViewChart({ data, markers = [] }: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "rgba(255, 255, 255, 0.7)",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.05)" },
        horzLines: { color: "rgba(255, 255, 255, 0.05)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
      },
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        timeVisible: true,
      },
      autoSize: true,
    });

    chartRef.current = chart;

    // Add Candlestick Series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    seriesRef.current = candleSeries;

    // Set Data
    if (data && data.length > 0) {
      // Sort and clean data for lightweight-charts
      const cleanData = data
        .filter(d => d.time) // Ensure time exists
        .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime())
        .map(d => ({
          time: d.time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));
      
      // Eliminate duplicates
      const uniqueData = [];
      let lastTime = "";
      for (const d of cleanData) {
        if (d.time !== lastTime) {
          uniqueData.push(d);
          lastTime = d.time;
        }
      }

      candleSeries.setData(uniqueData as any);
      
      // Set Markers
      if (markers && markers.length > 0) {
        // Sort markers by time
        const sortedMarkers = [...markers].sort((a, b) => 
          new Date(a.time).getTime() - new Date(b.time).getTime()
        );
        createSeriesMarkers(candleSeries as any, sortedMarkers as any);
      }

      chart.timeScale().fitContent();
    }

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data, markers]);

  return <div ref={chartContainerRef} className="w-full h-full min-h-[400px]" />;
}
