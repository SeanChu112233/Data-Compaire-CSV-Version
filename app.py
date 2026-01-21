import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO
import base64

# 设置matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义固定的对比颜色（可根据文件数量扩展）
DEFAULT_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def load_csv_file(uploaded_file):
    """加载CSV文件，兼容多种编码格式"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for encoding in encodings:
        try:
            string_data = uploaded_file.getvalue().decode(encoding)
            df = pd.read_csv(StringIO(string_data))
            # 去除空行和全空列
            df = df.dropna(how='all').dropna(axis=1, how='all')
            return df
        except UnicodeDecodeError:
            continue
    st.error(f"文件 {uploaded_file.name} 编码格式不支持，请检查文件")
    return None

def generate_time_series_plot(df_list, file_names, param_name, x_col):
    """
    生成指定参数的时间序列对比图
    :param df_list: 数据框列表
    :param file_names: 文件名列表
    :param param_name: 要对比的参数名
    :param x_col: X轴列名（时间列）
    :return: matplotlib figure对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 为每个文件绘制曲线（固定颜色）
    for idx, (df, file_name) in enumerate(zip(df_list, file_names)):
        # 获取数据（确保X轴和参数列存在）
        if x_col not in df.columns or param_name not in df.columns:
            st.warning(f"文件 {file_name} 缺少 {x_col} 或 {param_name} 列，跳过该文件")
            continue
        
        x_data = df[x_col].values
        y_data = df[param_name].values
        
        # 绘制曲线（使用固定颜色，超出默认颜色则循环）
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        ax.plot(x_data, y_data, 
                label=f"{file_name}", 
                color=color, 
                linewidth=2,
                alpha=0.8)
    
    # 设置图表样式
    ax.set_title(f'{param_name} 多文件时间序列对比', fontsize=14, pad=20)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(param_name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10, loc='best')
    
    # 调整布局
    plt.tight_layout()
    return fig

def get_download_link(fig, param_name, format='png'):
    """生成图表下载链接"""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/{format};base64,{b64}" download="{param_name}_对比图.{format}">下载{param_name}对比图</a>'

def main():
    st.set_page_config(page_title="多CSV时间序列对比", layout="wide")
    st.title("多CSV文件参数时间序列对比工具")
    st.markdown("### 上传说明")
    st.write("请上传**列名完全一致**的多个CSV文件，第一列默认为时间轴（X轴），其余列为对比参数")
    
    # 1. 多文件上传
    uploaded_files = st.file_uploader(
        "选择多个CSV文件",
        type="csv",
        accept_multiple_files=True,
        help="请确保所有文件的列名完全一致"
    )
    
    if not uploaded_files:
        st.info("请先上传至少2个CSV文件进行对比")
        return
    
    # 2. 加载所有文件并验证
    df_list = []
    file_names = []
    for file in uploaded_files:
        df = load_csv_file(file)
        if df is not None and not df.empty:
            df_list.append(df)
            file_names.append(file.name.split('.')[0])  # 去除文件后缀
    
    if len(df_list) < 2:
        st.error("有效文件数量不足2个，请检查文件内容")
        return
    
    # 3. 验证列名一致性
    first_columns = df_list[0].columns.tolist()
    for idx, df in enumerate(df_list[1:], 1):
        if df.columns.tolist() != first_columns:
            st.warning(f"文件 {file_names[idx]} 的列名与第一个文件不一致，请检查")
            st.write(f"第一个文件列名：{first_columns}")
            st.write(f"{file_names[idx]} 列名：{df.columns.tolist()}")
    
    # 4. 选择对比参数
    st.markdown("---")
    st.subheader("参数选择")
    param_options = first_columns[1:]  # 排除第一列（时间列）
    if not param_options:
        st.error("文件仅包含一列数据，无可用对比参数")
        return
    
    selected_param = st.selectbox(
        "选择要对比的参数",
        options=param_options,
        index=0,
        help="选择需要绘制时间序列的参数"
    )
    
    # 5. 生成并显示图表
    st.markdown("---")
    st.subheader("对比图表")
    x_col = first_columns[0]  # 第一列作为X轴（时间列）
    fig = generate_time_series_plot(df_list, file_names, selected_param, x_col)
    
    # 显示图表
    st.pyplot(fig)
    
    # 6. 下载链接
    st.markdown(get_download_link(fig, selected_param), unsafe_allow_html=True)
    
    # 7. 数据预览
    st.markdown("---")
    st.subheader("数据预览")
    tab_list = st.tabs(file_names)
    for idx, tab in enumerate(tab_list):
        with tab:
            st.dataframe(df_list[idx], use_container_width=True)

if __name__ == "__main__":
    main()
