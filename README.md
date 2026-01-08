# Agentic Pricing: Decision-Centric Reinforcement Learning for Dynamic Pricing

## Overview

Most pricing projects stop at prediction—forecasting demand or revenue.  
This project treats pricing as a sequential decision-making problem where a system must act, observe outcomes, and adapt over time.

The objective is not accuracy, but long-term profit optimization under competition and inventory constraints.

---

## Problem Statement

Real-world pricing is influenced by:
- Price elasticity of demand
- Competitor price changes
- Finite inventory
- Long-term consequences of short-term decisions

Static pricing and one-shot prediction models fail to capture these dynamics.

This project models pricing as a decision process where the system learns what price to set at each step to maximize cumulative profit.

---

## System Design

The system consists of three core components:

### Pricing Environment
A custom simulation models:
- Demand response to price
- Competitor price fluctuations
- Inventory depletion
- Profit as the reward signal

This formulation converts pricing into a Markov Decision Process.

### Baseline Strategy
A fixed-price strategy acts as a business control:
- Same price at every step
- No adaptation
- Measures average profit and stockout behavior

This provides a realistic benchmark.

### Learning Agent
A Q-learning agent:
- Observes price, competitor price, inventory, and time
- Discretizes the state space
- Learns pricing actions through interaction
- Optimizes cumulative profit rather than short-term gain

---

## Project Structure

AgenticPricing/
├── env/
│   └── pricing_env.py
├── baselines/
│   └── fixed_price.py
├── agents/
│   └── q_learning_agent.py
├── plots/
│   └── learning_curve.png
├── run_experiment.py
├── requirements.txt
└── README.md

---

## How to Run

Install dependencies:

Run the experiment:

This executes the baseline strategy, trains the learning agent, evaluates both, and saves a learning curve plot.

---

## Results

The system reports:
- Average profit of a fixed-price baseline
- Average profit of the trained agent
- A learning curve showing agent behavior over time

Initial runs may show the agent underperforming the baseline, highlighting exploration–exploitation tradeoffs. With tuning, the agent adapts pricing decisions based on demand and competition.

---

## Why This Project Is Different

- Focuses on decisions, not predictions
- Uses profit as the primary metric
- Includes a business control for comparison
- Exposes real reinforcement learning challenges
- Reflects production-style pricing problems

---

## Key Concepts

- Reinforcement Learning
- Sequential Decision Making
- Price Elasticity
- Baseline vs Agent Evaluation
- Exploration vs Exploitation
- Decision-Centric Data Science

---

## Notes

The project is intentionally lightweight:
- Minimal dependencies
- Clear logic over abstraction
- Emphasis on decision quality and business impact
