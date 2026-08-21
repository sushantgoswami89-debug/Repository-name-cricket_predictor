# CricketBaba Model Contract

## Runs Model

Model: LightGBM Regressor

Expected Features (22)

1. over
2. score_before_over
3. wkts_down_before_over
4. balls_faced_before_over
5. venue_avg_score
6. bat_career_overs_faced
7. bat_hist_avg_runs_per_over
8. bat_hist_wicket_rate
9. bowl_career_overs_bowled
10. bowl_hist_avg_runs_conceded
11. bowl_hist_wicket_rate
12. h2h_overs
13. h2h_avg_runs
14. bat_vs_bowltype_avg_runs
15. bat_vs_bowltype_wicket_rate
16. bowl_phase_avg_runs
17. phase
18. batsman_style
19. batsman_class
20. bowler_type
21. bowler_quality
22. pitch_type

---

## Feature Sources

### LiveMatchState

- over
- score_before_over
- wkts_down_before_over
- balls_faced_before_over

### MatchContext

- pitch_type

### Venue Database

- venue_avg_score

### Batter Statistics

- bat_career_overs_faced
- bat_hist_avg_runs_per_over
- bat_hist_wicket_rate
- bat_vs_bowltype_avg_runs
- bat_vs_bowltype_wicket_rate
- batsman_style
- batsman_class

### Bowler Statistics

- bowl_career_overs_bowled
- bowl_hist_avg_runs_conceded
- bowl_hist_wicket_rate
- bowl_phase_avg_runs
- bowler_type
- bowler_quality

### Head-to-Head

- h2h_overs
- h2h_avg_runs

### Derived

- phase