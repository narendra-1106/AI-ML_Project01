/**
 * IPL Analytics Dashboard - Chart.js Visualizations
 * Fetches real historical data from /api/dashboard-data
 */

document.addEventListener("DOMContentLoaded", function () {
    fetch("/api/dashboard-data")
        .then((response) => response.json())
        .then((data) => {
            renderDashboardCharts(data);
        })
        .catch((error) => {
            console.error("Error loading dashboard data:", error);
        });
});

function renderDashboardCharts(data) {
    // Brand IPL Color Palette
    const colors = [
        "#1e40af", "#3b82f6", "#0284c7", "#06b6d4",
        "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#ec4899", "#6366f1", "#14b8a6", "#f97316",
        "#64748b", "#84cc16"
    ];

    // 1. Team Wins Chart (Bar)
    const ctxWins = document.getElementById("teamWinsChart").getContext("2d");
    new Chart(ctxWins, {
        type: "bar",
        data: {
            labels: data.team_names,
            datasets: [{
                label: "Total Matches Won",
                data: data.team_wins,
                backgroundColor: colors,
                borderRadius: 6,
                borderWidth: 1,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Wins: ${context.raw} matches`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: "Wins Count" },
                    grid: { color: "#f1f5f9" }
                },
                x: {
                    ticks: {
                        autoSkip: false,
                        maxRotation: 45,
                        minRotation: 30,
                        font: { size: 10 }
                    },
                    grid: { display: false }
                }
            }
        }
    });

    // 2. Matches per Season Chart (Line)
    const ctxSeason = document.getElementById("seasonChart").getContext("2d");
    new Chart(ctxSeason, {
        type: "line",
        data: {
            labels: data.seasons,
            datasets: [{
                label: "Matches Scheduled",
                data: data.matches_per_season,
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.15)",
                fill: true,
                tension: 0.3,
                pointBackgroundColor: "#1e40af",
                pointBorderColor: "#ffffff",
                pointHoverRadius: 7,
                pointRadius: 4,
                borderWidth: 2.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Season ${context.label}: ${context.raw} Matches`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 40,
                    title: { display: true, text: "Number of Matches" },
                    grid: { color: "#f1f5f9" }
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 10 } }
                }
            }
        }
    });

    // 3. Toss Outcome Impact Chart (Doughnut)
    const ctxTossOutcome = document.getElementById("tossOutcomeChart").getContext("2d");
    const tossWinPct = data.toss_impact.toss_winner_won_match_pct;
    const tossLosePct = (100 - tossWinPct).toFixed(2);

    new Chart(ctxTossOutcome, {
        type: "doughnut",
        data: {
            labels: ["Toss Winner Won Match", "Toss Winner Lost Match"],
            datasets: [{
                data: [tossWinPct, tossLosePct],
                backgroundColor: ["#10b981", "#ef4444"],
                borderWidth: 2,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom" },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            },
            cutout: "65%"
        }
    });

    // 4. Toss Decision Distribution (Pie)
    const ctxTossDecision = document.getElementById("tossDecisionChart").getContext("2d");
    const fieldCount = data.toss_impact.decision_breakdown.field || 800;
    const batCount = data.toss_impact.decision_breakdown.bat || 412;

    new Chart(ctxTossDecision, {
        type: "pie",
        data: {
            labels: ["Chose to Field (Bowling)", "Chose to Bat (Setting Target)"],
            datasets: [{
                data: [fieldCount, batCount],
                backgroundColor: ["#0284c7", "#f59e0b"],
                borderWidth: 2,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom" },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const total = fieldCount + batCount;
                            const pct = ((context.raw / total) * 100).toFixed(1);
                            return `${context.label}: ${context.raw} times (${pct}%)`;
                        }
                    }
                }
            }
        }
    });

    // 5. Franchise Win Percentage Chart (Bar)
    const ctxWinPct = document.getElementById("winPctChart").getContext("2d");
    new Chart(ctxWinPct, {
        type: "bar",
        data: {
            labels: data.team_names,
            datasets: [{
                label: "Win Percentage (%)",
                data: data.team_win_pcts,
                backgroundColor: "#4f46e5",
                borderRadius: 5,
                borderWidth: 1,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Win Rate: ${context.raw}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 70,
                    title: { display: true, text: "Win %" },
                    grid: { color: "#f1f5f9" }
                },
                x: {
                    ticks: {
                        autoSkip: false,
                        maxRotation: 45,
                        minRotation: 30,
                        font: { size: 10 }
                    },
                    grid: { display: false }
                }
            }
        }
    });

    // 6. Top Venues Chart (Horizontal Bar)
    const ctxVenue = document.getElementById("venueChart").getContext("2d");
    new Chart(ctxVenue, {
        type: "bar",
        data: {
            labels: data.top_venues_labels.slice(0, 8),
            datasets: [{
                label: "Matches Hosted",
                data: data.top_venues_counts.slice(0, 8),
                backgroundColor: "#0ea5e9",
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: { display: true, text: "Matches Hosted" },
                    grid: { color: "#f1f5f9" }
                },
                y: {
                    ticks: { font: { size: 11 } },
                    grid: { display: false }
                }
            }
        }
    });
}
