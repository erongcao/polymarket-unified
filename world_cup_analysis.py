#!/usr/bin/env python3
"""
世界杯冠军竞猜市场分析
使用严谨理论框架分析实际Polymarket数据
"""

import sys
sys.path.insert(0, '/Users/yirongcao/.openclaw/workspace/skills/polymarket-unified')

from polymarket_analysis_suite import RigorousPolymarketAnalyzer
import numpy as np

# 2026世界杯冠军竞猜市场数据（基于Polymarket实际结构）
# 数据来源：模拟真实市场结构（阿根廷、法国、巴西等热门球队）

world_cup_event = {
    'title': '2026 FIFA World Cup Winner',
    'slug': '2026-fifa-world-cup-winner',
    'category': 'sports',
    'volume': 2850000,  # $2.85M交易量
    'num_outcomes': 32,  # 32支参赛球队
    'outcomes': [
        {'name': 'Argentina', 'probability': 0.155, 'odds_implied': 0.120},
        {'name': 'France', 'probability': 0.142, 'odds_implied': 0.135},
        {'name': 'Brazil', 'probability': 0.128, 'odds_implied': 0.115},
        {'name': 'Germany', 'probability': 0.095, 'odds_implied': 0.080},
        {'name': 'Spain', 'probability': 0.088, 'odds_implied': 0.075},
        {'name': 'England', 'probability': 0.082, 'odds_implied': 0.070},
        {'name': 'Portugal', 'probability': 0.065, 'odds_implied': 0.055},
        {'name': 'Netherlands', 'probability': 0.045, 'odds_implied': 0.040},
        {'name': 'Italy', 'probability': 0.038, 'odds_implied': 0.035},
        {'name': 'Belgium', 'probability': 0.032, 'odds_implied': 0.030},
        # 其余22支球队概率较低，合并为"Other"
        {'name': 'Other', 'probability': 0.130, 'odds_implied': 0.365}
    ],
    'resolution_date': '2026-07-19',
    'market_maker_type': 'logarithmic',
    'liquidity_parameter': 500  # b = 500
}

print("=" * 80)
print("2026 FIFA WORLD CUP CHAMPION - RIGOROUS ANALYSIS")
print("=" * 80)
print(f"Market: {world_cup_event['title']}")
print(f"Volume: ${world_cup_event['volume']:,}")
print(f"Outcomes: {world_cup_event['num_outcomes']} teams")
print(f"Resolution: {world_cup_event['resolution_date']}")
print()

# 创建分析器
analyzer = RigorousPolymarketAnalyzer(world_cup_event)

# 1. HARA流动性分析（不同风险参数）
print("1. HARA UTILITY-BASED LIQUIDITY ANALYSIS (Chen & Pennock 2007)")
print("-" * 80)
print("Analyzing liquidity-loss tradeoff for different risk preferences...")

# 简化：对32支球队使用聚合表示（前10 + Other）
simplified_event = {
    'title': world_cup_event['title'],
    'outcomes': world_cup_event['outcomes'][:11],  # 前10 + Other
    'volume': world_cup_event['volume'],
    'category': 'sports'
}

analyzer = RigorousPolymarketAnalyzer(simplified_event)

# 只分析2种代表性gamma值（计算成本考虑）
hara_results = analyzer.analyze_hara_liquidity(gamma_values=[1.0, 2.0])

for gamma_key, data in hara_results.items():
    if 'error' not in data:
        print(f"\n  {gamma_key}:")
        print(f"    Max loss bound: ${data['max_loss']:.2f}")
        print(f"    Liquidity focus: {data['focus']}")
        print(f"    Sample prices: {[f'{p:.3f}' for p in data['prices'][:3]]}...")

# 2. 模拟交易者贡献分析（MC Shapley）
print("\n" + "=" * 80)
print("2. TRADER CONTRIBUTION ANALYSIS (Monte Carlo Shapley)")
print("-" * 80)
print("Simulating trade history for major teams...")

# 模拟真实交易流
trade_history = [
    # 大额投注阿根廷（看好卫冕冠军）
    {'trader_id': 0, 'outcome': 0, 'shares': 150000, 'price': 0.145},  # Argentina
    {'trader_id': 0, 'outcome': 0, 'shares': 50000, 'price': 0.152},
    # 法国对冲
    {'trader_id': 1, 'outcome': 1, 'shares': 120000, 'price': 0.138},  # France
    {'trader_id': 2, 'outcome': 1, 'shares': 80000, 'price': 0.140},
    # 巴西投注
    {'trader_id': 3, 'outcome': 2, 'shares': 100000, 'price': 0.125},  # Brazil
    # 套利者调整
    {'trader_id': 4, 'outcome': 0, 'shares': -30000, 'price': 0.155},  # 卖出阿根廷
    {'trader_id': 4, 'outcome': 1, 'shares': 40000, 'price': 0.142},   # 买入法国
    # 德国冷门投注
    {'trader_id': 5, 'outcome': 3, 'shares': 60000, 'price': 0.090},   # Germany
    # 英格兰投注
    {'trader_id': 6, 'outcome': 5, 'shares': 70000, 'price': 0.080},   # England
    # 信息聚合者大额均衡
    {'trader_id': 7, 'outcome': 0, 'shares': 200000, 'price': 0.150},
    {'trader_id': 7, 'outcome': 1, 'shares': 180000, 'price': 0.140},
    {'trader_id': 7, 'outcome': 2, 'shares': 160000, 'price': 0.125},
]

# 样本数减少以控制计算时间
shapley_results = analyzer.analyze_trader_contributions(trade_history, n_samples=500)

print(f"\n  Top 3 Information Contributors (Shapley Value):")
for i, (trader_id, value) in enumerate(shapley_results['key_traders'][:3], 1):
    print(f"    {i}. Trader {trader_id}: {value:.6f}")

print(f"\n  Information Concentration:")
print(f"    Gini coefficient: {shapley_results['concentration']['gini']:.3f}")
print(f"    Herfindahl index: {shapley_results['concentration']['hhi']:.3f}")
print(f"    Risk flag: {shapley_results['concentration']['risk_flag']}")

# 3. 均衡学习分析
print("\n" + "=" * 80)
print("3. EQUILIBRIUM LEARNING ANALYSIS (Fictitious Play)")
print("-" * 80)
print("Simulating trader learning dynamics...")

# 真实分布（基于历史世界杯胜率）
true_distribution = np.array([
    0.12,  # Argentina (卫冕冠军，主场美洲，高)
    0.15,  # France (近年最强，但世界杯魔咒)
    0.13,  # Brazil (南美强队，但26年北美世界杯)
    0.08,  # Germany (重建期，较低)
    0.07,  # Spain
    0.06,  # England
    0.04,  # Portugal
    0.03,  # Netherlands
    0.02,  # Italy
    0.02,  # Belgium
    0.28   # Other (冷门球队合计)
])

# 归一化
true_distribution = true_distribution / np.sum(true_distribution)

eq_results = analyzer.analyze_equilibrium_learning(
    true_distribution, 
    n_iterations=200  # 减少迭代以控制时间
)

print(f"\n  Aggregate Prediction (Equilibrium):")
agg_pred = eq_results['aggregate_prediction']
print(f"    Argentina: {agg_pred[0]:.3f} (True: {true_distribution[0]:.3f})")
print(f"    France: {agg_pred[1]:.3f} (True: {true_distribution[1]:.3f})")
print(f"    Brazil: {agg_pred[2]:.3f} (True: {true_distribution[2]:.3f})")
print(f"    Germany: {agg_pred[3]:.3f} (True: {true_distribution[3]:.3f})")

print(f"\n  Prediction Error (L2): {eq_results['prediction_error']:.4f}")
print(f"  Converged: {eq_results['converged']}")

# 4. 执行性偏差检测（简化）
print("\n" + "=" * 80)
print("4. PERFORMATIVE BIAS ANALYSIS (Oesterheld et al. 2023)")
print("-" * 80)
print("Checking if market predictions influence outcomes...")

# 模拟价格历史（小组赛抽签后波动）
price_history = [
    0.120,  # 初始（抽签前）
    0.135,  # 抽签后（阿根廷好签）
    0.142,  # 媒体报道增加
    0.155,  # 梅西状态传闻
    0.148,  # 回归理性
    0.152,  # 当前
]

# 模拟实际胜率变化（受市场影响）
outcome_proxy = [
    0.115,  # 基准
    0.118,  # 轻微影响（媒体关注）
    0.122,  # 更多资源投入
    0.128,  # 对手研究
    0.125,  # 调整
    0.120,  # 回归
]

perf_results = analyzer.performative_bias_check(price_history, outcome_proxy)

print(f"\n  Price-Outcome Correlation: {perf_results['correlation']:.3f}")
print(f"  Bias Level: {perf_results['bias_level']}")
print(f"  Recommendation: {perf_results['recommendation']}")

# 生成完整报告
print("\n" + "=" * 80)
print(analyzer.full_report())
