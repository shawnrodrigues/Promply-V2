"use client";

import { ComponentProps, useCallback, useEffect, useRef } from "react";

import { cn } from "../../lib/utils";

type DotLoaderProps = {
    frames: number[][];
    dotClassName?: string;
    isPlaying?: boolean;
    duration?: number;
    repeatCount?: number;
    onComplete?: () => void;
    theme?: "default" | "cyber" | "emerald" | "purple";
} & ComponentProps<"div">;

export const DotLoader = ({
    frames,
    isPlaying = true,
    duration = 100,
    dotClassName,
    className,
    repeatCount = -1,
    onComplete,
    theme = "default",
    ...props
}: DotLoaderProps) => {
    const gridRef = useRef<HTMLDivElement>(null);
    const currentIndex = useRef(0);
    const repeats = useRef(0);
    const interval = useRef<NodeJS.Timeout>(null);

    const getThemeClasses = useCallback(() => {
        switch (theme) {
            case "cyber":
                return {
                    inactive: "bg-slate-800/30 border border-cyan-500/20",
                    active: "bg-gradient-to-br from-cyan-400 to-blue-500 border border-cyan-300/50 shadow-lg shadow-cyan-500/50 scale-110",
                };
            case "emerald":
                return {
                    inactive: "bg-slate-800/30 border border-emerald-500/20",
                    active: "bg-gradient-to-br from-emerald-400 to-green-500 border border-emerald-300/50 shadow-lg shadow-emerald-500/50 scale-110",
                };
            case "purple":
                return {
                    inactive: "bg-slate-800/30 border border-purple-500/20",
                    active: "bg-gradient-to-br from-purple-400 to-violet-500 border border-purple-300/50 shadow-lg shadow-purple-500/50 scale-110",
                };
            default:
                return {
                    inactive: "bg-slate-700/50",
                    active: "bg-blue-500 scale-110 shadow-lg shadow-blue-500/50",
                };
        }
    }, [theme]);

    const applyFrameToDots = useCallback(
        (dots: HTMLDivElement[], frameIndex: number) => {
            const frame = frames[frameIndex];
            if (!frame) return;

            const themeClasses = getThemeClasses();

            dots.forEach((dot, index) => {
                // Remove all theme classes first
                dot.className = dot.className.replace(/bg-\S+|border\S*|shadow\S*|scale\S*/g, "").trim();

                if (frame.includes(index)) {
                    dot.className = cn(dot.className, themeClasses.active, "transition-all duration-200 ease-out");
                } else {
                    dot.className = cn(dot.className, themeClasses.inactive, "transition-all duration-200 ease-out");
                }
            });
        },
        [frames, getThemeClasses],
    );

    useEffect(() => {
        currentIndex.current = 0;
        repeats.current = 0;
    }, [frames]);

    useEffect(() => {
        if (isPlaying) {
            if (currentIndex.current >= frames.length) {
                currentIndex.current = 0;
            }
            const dotElements = gridRef.current?.children;
            if (!dotElements) return;
            const dots = Array.from(dotElements) as HTMLDivElement[];
            interval.current = setInterval(() => {
                applyFrameToDots(dots, currentIndex.current);
                if (currentIndex.current + 1 >= frames.length) {
                    if (repeatCount != -1 && repeats.current + 1 >= repeatCount) {
                        clearInterval(interval.current!);
                        onComplete?.();
                    }
                    repeats.current++;
                }
                currentIndex.current = (currentIndex.current + 1) % frames.length;
            }, duration);
        } else {
            if (interval.current) clearInterval(interval.current);
        }

        return () => {
            if (interval.current) clearInterval(interval.current);
        };
    }, [frames, isPlaying, applyFrameToDots, duration, repeatCount, onComplete]);

    return (
        <div
            {...props}
            ref={gridRef}
            className={cn("grid w-fit grid-cols-7 gap-1 p-2 rounded-xl bg-slate-900/20 backdrop-blur-sm", className)}
        >
            {Array.from({ length: 49 }).map((_, i) => (
                <div
                    key={i}
                    className={cn(
                        "h-2 w-2 rounded-full transition-all duration-200 ease-out",
                        getThemeClasses().inactive,
                        dotClassName,
                    )}
                />
            ))}
        </div>
    );
};
