import React from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Area, AreaChart, ReferenceLine
} from 'recharts';
import { Activity } from 'lucide-react';
import './ProbabilityChart.css';

const ProbabilityChart = ({ data, currentTime, totalFrames, onSeek }) => {
    // Format data for the chart
    const chartData = data?.map(p => ({
        frame: p.frame,
        prob: p.probabilities.FAKE,
        time: p.frame / (totalFrames / (data.length * 2)) // Rough estimation
    })) || [];

    const handleChartClick = (e) => {
        if (e && e.activePayload && onSeek) {
            const frameIndex = e.activePayload[0].payload.frame;
            onSeek(frameIndex);
        }
    };

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            return (
                <div className="custom-tooltip forensic-panel">
                    <p className="label">{`FRAME: `}<span className="mono-data">{payload[0].payload.frame}</span></p>
                    <p className="value">{`FAKE PROB: ${(payload[0].value * 100).toFixed(2)}%`}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="chart-container forensic-panel">
            <div className="chart-header">
                <h3>
                    <Activity size={18} />
                    Neural Pattern Analysis
                </h3>
                <p>Frame-by-frame probability distribution matrix</p>
            </div>

            <div className="chart-body">
                <ResponsiveContainer width="100%" height={260}>
                    <AreaChart data={chartData} onClick={handleChartClick} style={{ cursor: 'crosshair' }}>
                        <defs>
                            <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="var(--accent-red)" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="var(--accent-red)" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.08)" vertical={true} horizontal={true} />
                        <XAxis
                            dataKey="frame"
                            stroke="var(--text-secondary)"
                            fontSize={11}
                            fontFamily="var(--font-mono)"
                            tickLine={false}
                            axisLine={false}
                            dy={10}
                        />
                        <YAxis
                            stroke="var(--text-secondary)"
                            fontSize={11}
                            fontFamily="var(--font-mono)"
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                            dx={-10}
                        />
                        <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.2)', strokeWidth: 1 }} />
                        <Area
                            type="linear"
                            dataKey="prob"
                            stroke="var(--accent-red)"
                            strokeWidth={1.5}
                            fillOpacity={1}
                            fill="url(#colorProb)"
                            activeDot={{ r: 4, stroke: 'var(--bg-panel)', strokeWidth: 1, fill: 'var(--accent-red)' }}
                        />
                        {/* Playhead representation */}
                        {currentTime !== undefined && (
                            <ReferenceLine
                                x={Math.round(currentTime)}
                                stroke="var(--accent-blue)"
                                strokeWidth={1}
                                strokeDasharray="4 2"
                                isFront={true}
                            />
                        )}
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default ProbabilityChart;
