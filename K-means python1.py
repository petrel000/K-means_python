import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 全局配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(
    page_title="患者心血管风险分群评估系统",
    page_icon="🏥",
    layout="wide"
)

# 初始化会话状态，保存关键参数
if 'k_selected' not in st.session_state:
    st.session_state.k_selected = 3
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'df_result' not in st.session_state:
    st.session_state.df_result = None

# 标题与说明
st.title("🏥 患者心血管风险分群评估系统")
st.markdown("基于K-Means无监督学习的心血管疾病与代谢综合征风险筛查")

# 侧边栏配置
st.sidebar.header("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传患者数据(CSV)", type=['csv'])

st.sidebar.header("⚙️ 参数设置")
k_value = st.sidebar.slider("聚类数量 K", 2, 10, st.session_state.k_selected)
auto_k = st.sidebar.checkbox("自动选择最佳K值", value=False)  # 默认关闭自动选K，避免覆盖手动设置

# 数据加载与验证
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 数据预览")
    st.write(df.head())
    
    # 特征列定义与验证
    feature_cols = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'blood_sugar', 'bmi', 'cholesterol']
    available_features = [f for f in feature_cols if f in df.columns]
    
    if len(available_features) < 2:
        st.error("数据列名不匹配！需要包含至少2个以下列：age, systolic_bp, diastolic_bp, heart_rate, blood_sugar, bmi, cholesterol")
    else:
        # 特征选择（关联会话状态）
        st.session_state.selected_features = st.sidebar.multiselect(
            "选择用于聚类的特征",
            available_features,
            default=st.session_state.selected_features if st.session_state.selected_features else available_features[:4]
        )
        
        # 分析按钮逻辑
        if st.sidebar.button("🚀 开始分析"):
            with st.spinner("分析中..."):
                # 1. 数据预处理
                X = df[st.session_state.selected_features]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state.scaler = scaler  # 保存标准化器到会话状态
                
                # 2. K值选择逻辑（优先手动，自动为辅）
                if auto_k:
                    best_k, best_score = 2, -1
                    # 自动选K范围：2到10（与滑块范围一致）
                    for k_candidate in range(2, 11):
                        km = KMeans(n_clusters=k_candidate, random_state=42, n_init=10)
                        labels = km.fit_predict(X_scaled)
                        score = silhouette_score(X_scaled, labels)
                        if score > best_score:
                            best_k, best_score = k_candidate, score
                    final_k = best_k
                    st.info(f"自动选择最佳K值：K = {final_k}（轮廓系数：{best_score:.3f}）")
                else:
                    final_k = k_value
                    st.info(f"使用手动设置K值：K = {final_k}")
                
                # 更新会话状态的K值
                st.session_state.k_selected = final_k
                
                # 3. K-Means聚类
                kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
                df['cluster'] = kmeans.fit_predict(X_scaled)
                st.session_state.model = kmeans  # 保存模型到会话状态
                
                # 4. 群体特征描述函数（优化判断逻辑）
                def describe_cluster(cid):
                    center = kmeans.cluster_centers_[cid]
                    center_original = scaler.inverse_transform([center])[0]
                    features = []
                    feat_thresholds = {
                        'age': (60, '高龄'),
                        'systolic_bp': (140, '收缩压偏高'),
                        'diastolic_bp': (90, '舒张压偏高'),
                        'blood_sugar': (126, '血糖偏高'),
                        'bmi': (28, '体重超标'),
                        'cholesterol': (240, '胆固醇偏高'),
                        'heart_rate': (100, '心率偏快')
                    }
                    for i, feat in enumerate(st.session_state.selected_features):
                        val = center_original[i]
                        if feat in feat_thresholds and val > feat_thresholds[feat][0]:
                            features.append(feat_thresholds[feat][1])
                    return "、".join(features) + "群体" if features else "指标正常群体"
                
                # 5. 风险等级评估（优化评分逻辑）
                def assess_risk(cid):
                    # 基于聚类中心与阈值的偏离程度计算风险
                    center = kmeans.cluster_centers_[cid]
                    # 计算标准化后中心的均值（正值越大风险越高）
                    risk_score = np.mean(center)
                    if risk_score > 0.5:
                        return '高风险'
                    elif risk_score > -0.3:
                        return '中风险'
                    else:
                        return '低风险'
                
                # 6. 生成结果数据
                df['cluster_type'] = df['cluster'].apply(describe_cluster)
                df['risk_level'] = df['cluster'].apply(assess_risk)
                st.session_state.df_result = df  # 保存结果到会话状态
                
                # 7. PCA降维可视化
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                df['pca1'] = X_pca[:, 0]
                df['pca2'] = X_pca[:, 1]
                
                # 8. 展示群体统计
                st.write("### 📊 各群体统计特征")
                cluster_stats = []
                for c in range(final_k):
                    group = df[df['cluster'] == c]
                    stats = {
                        '群体ID': c,
                        '群体特征': describe_cluster(c),
                        '风险等级': assess_risk(c),
                        '人数': len(group),
                        '占比': f"{len(group)/len(df)*100:.1f}%",
                        **{f'平均{feat}': round(group[feat].mean(), 1) if feat in group.columns else '-' 
                           for feat in st.session_state.selected_features}
                    }
                    cluster_stats.append(stats)
                st.table(pd.DataFrame(cluster_stats))
                
                # 9. PCA可视化
                st.write("### 📈 PCA降维聚类视图")
                fig, ax = plt.subplots(figsize=(12, 7))
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
                for c in range(final_k):
                    mask = df['cluster'] == c
                    ax.scatter(df.loc[mask, 'pca1'], df.loc[mask, 'pca2'], 
                              c=colors[c], label=f'群体{c} ({describe_cluster(c)})', alpha=0.7, s=60)
                ax.set_xlabel(f'PC1 (解释方差：{pca.explained_variance_ratio_[0]:.2f})', fontsize=12)
                ax.set_ylabel(f'PC2 (解释方差：{pca.explained_variance_ratio_[1]:.2f})', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # 10. 患者明细
                st.write("### 👥 患者分群明细")
                display_cols = ['patient_id'] + st.session_state.selected_features + ['cluster', 'cluster_type', 'risk_level']
                available_display = [c for c in display_cols if c in df.columns]
                st.dataframe(df[available_display].head(100))

    # 新患者预测模块（独立于分析按钮，依赖会话状态）
    if st.session_state.model is not None and st.session_state.scaler is not None:
        st.write("### 🔍 新患者风险评估")
        
        # 动态生成输入框（根据选择的特征）
        input_cols = st.columns(len(st.session_state.selected_features))
        new_patient_data = {}
        
        # 特征默认值配置（更合理的医学默认值）
        default_values = {
            'age': 50.0, 'systolic_bp': 120.0, 'diastolic_bp': 80.0,
            'heart_rate': 75.0, 'blood_sugar': 100.0, 'bmi': 24.0,
            'cholesterol': 200.0
        }
        
        # 生成对应特征的输入框
        for idx, feat in enumerate(st.session_state.selected_features):
            with input_cols[idx]:
                new_patient_data[feat] = st.number_input(
                    feat.replace('_', ' ').title(),
                    value=default_values.get(feat, 0.0),
                    key=f"new_{feat}"
                )
        
        # 预测按钮
        if st.button("预测风险"):
            # 构造预测数据数组
            new_data_array = np.array([[new_patient_data[feat] for feat in st.session_state.selected_features]])
            # 使用保存的标准化器做转换（避免数据泄露）
            new_data_scaled = st.session_state.scaler.transform(new_data_array)
            # 预测聚类
            pred_cluster = st.session_state.model.predict(new_data_scaled)[0]
            
            # 生成预测结果
            def get_cluster_desc(cid):
                center = st.session_state.model.cluster_centers_[cid]
                center_original = st.session_state.scaler.inverse_transform([center])[0]
                features = []
                feat_thresholds = {
                    'age': (60, '高龄'),
                    'systolic_bp': (140, '收缩压偏高'),
                    'diastolic_bp': (90, '舒张压偏高'),
                    'blood_sugar': (126, '血糖偏高'),
                    'bmi': (28, '体重超标'),
                    'cholesterol': (240, '胆固醇偏高'),
                    'heart_rate': (100, '心率偏快')
                }
                for i, feat in enumerate(st.session_state.selected_features):
                    val = center_original[i]
                    if feat in feat_thresholds and val > feat_thresholds[feat][0]:
                        features.append(feat_thresholds[feat][1])
                return "、".join(features) + "群体" if features else "指标正常群体"
            
            def get_risk_level(cid):
                risk_score = np.mean(st.session_state.model.cluster_centers_[cid])
                if risk_score > 0.5:
                    return '高风险'
                elif risk_score > -0.3:
                    return '中风险'
                else:
                    return '低风险'
            
            # 展示预测结果
            st.success(f"""
            ### 新患者风险预测结果
            - 所属聚类群体：**群体{pred_cluster}**
            - 群体特征描述：**{get_cluster_desc(pred_cluster)}**
            - 心血管风险等级：**{get_risk_level(pred_cluster)}**
            
            #### 输入指标对比群体均值
            | 特征 | 输入值 | 群体均值 |
            |------|--------|----------|
            {''.join([f'| {feat.replace('_', ' ').title()} | {new_patient_data[feat]:.1f} | {st.session_state.scaler.inverse_transform([st.session_state.model.cluster_centers_[pred_cluster]])[0][idx]:.1f} |\n' 
                     for idx, feat in enumerate(st.session_state.selected_features)])}
            """)
else:
    st.info("👈 请从左侧上传患者数据文件（CSV格式），支持的特征列：age, systolic_bp, diastolic_bp, heart_rate, blood_sugar, bmi, cholesterol")

# 底部说明
st.markdown("---")
st.markdown("""
**数据说明：**
- 本系统使用的医学指标包括：年龄、收缩压、舒张压、心率、空腹血糖、体重指数（BMI）、总胆固醇
- 风险判断参考标准（简化版，仅供算法演示）：
  - 收缩压 ≥140mmHg 或 舒张压 ≥90mmHg：参考《中国高血压防治指南》
  - 空腹血糖 ≥126mg/dL：参考美国糖尿病协会（ADA）标准
  - BMI ≥28：参考《中国成人超重和肥胖预防控制指南》
  - 总胆固醇 ≥240mg/dL：参考《中国成人血脂异常防治指南》

**免责声明：** 本系统为K-Means无监督学习算法演示项目，使用的风险判断标准为简化版，仅用于教学和研究目的。不构成医学诊断或治疗建议，实际医疗决策请咨询专业医师。
""")