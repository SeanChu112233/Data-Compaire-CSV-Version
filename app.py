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

def generate_time_series_plot(df_list, file_names, param_name, x_col, time_range, y_range):
    """
    生成指定参数的时间序列对比图（支持时间区间+纵坐标区间筛选）
    :param df_list: 数据框列表
    :param file_names: 文件名列表
    :param param_name: 要对比的参数名
    :param x_col: X轴列名（时间列）
    :param time_range: 时间区间 (start, end)
    :param y_range: 纵坐标区间 (y_min, y_max)
    :return: matplotlib figure对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 为每个文件绘制曲线（固定颜色）
    for idx, (df, file_name) in enumerate(zip(df_list, file_names)):
        # 获取数据（确保X轴和参数列存在）
        if x_col not in df.columns or param_name not in df.columns:
            st.warning(f"文件 {file_name} 缺少 {x_col} 或 {param_name} 列，跳过该文件")
            continue
        
        # 筛选指定时间区间的数据
        df_filtered = df[(df[x_col] >= time_range[0]) & (df[x_col] <= time_range[1])]
        if df_filtered.empty:
            st.warning(f"文件 {file_name} 在 [{time_range[0]}, {time_range[1]}] 区间内无数据")
            continue
        
        x_data = df_filtered[x_col].values
        y_data = df_filtered[param_name].values
        
        # 绘制曲线（使用固定颜色，超出默认颜色则循环）
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        ax.plot(x_data, y_data, 
                label=f"{file_name}", 
                color=color, 
                linewidth=2,
                alpha=0.8)
    
    # 设置纵坐标范围
    ax.set_ylim(y_range[0], y_range[1])
    
    # 设置图表样式
    ax.set_title(f'{param_name} 多文件时间序列对比（时间：{time_range[0]} ~ {time_range[1]} | 数值：{y_range[0]} ~ {y_range[1]}）', fontsize=14, pad=20)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(param_name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10, loc='best')
    
    # 调整布局
    plt.tight_layout()
    return fig

def get_download_link(fig, param_name, time_range, y_range, format='png'):
    """生成图表下载链接（包含时间区间和纵坐标区间信息）"""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    filename = f"{param_name}_对比图_时间{time_range[0]}-{time_range[1]}_数值{y_range[0]}-{y_range[1]}"
    return f'<a href="data:image/{format};base64,{b64}" download="{filename}.{format}">下载{param_name}对比图</a>'

def main():
    st.set_page_config(page_title="多CSV时间序列对比", layout="wide")
    st.title("多CSV文件参数时间序列对比工具")
    st.markdown("### 上传说明")
    st.write("请上传**列名完全一致**的多个CSV文件，第一列默认为时间轴（X轴），其余列为对比参数")
    
    # 初始化session state（用更稳定的方式实现复位）
    if 'reset_time' not in st.session_state:
        st.session_state.reset_time = False
    if 'reset_y' not in st.session_state:
        st.session_state.reset_y = False
    if 'default_time_range' not in st.session_state:
        st.session_state.default_time_range = (0, 0)
    if 'default_y_range' not in st.session_state:
        st.session_state.default_y_range = (0, 0)
    if 'current_param' not in st.session_state:
        st.session_state.current_param = ""
    
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
    
    # 处理参数选择和复位联动
    selected_param = st.selectbox(
        "选择要对比的参数",
        options=param_options,
        index=0 if st.session_state.current_param == "" else param_options.index(st.session_state.current_param),
        help="选择需要绘制时间序列的参数"
    )
    st.session_state.current_param = selected_param
    
    # 5. 时间区间选择（修复复位功能）
    st.markdown("---")
    st.subheader("时间区间筛选")
    x_col = first_columns[0]  # 第一列作为X轴（时间列）
    
    # 获取所有文件的时间范围，作为默认值
    all_x_values = []
    for df in df_list:
        if x_col in df.columns:
            all_x_values.extend(df[x_col].dropna().values)
    
    if not all_x_values:
        st.error("无法获取时间轴数据，请检查文件中是否有有效时间列")
        return
    
    min_x = min(all_x_values)
    max_x = max(all_x_values)
    st.session_state.default_time_range = (min_x, max_x)
    
    # 时间区间布局：起始时间 + 结束时间 + 复位按钮
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        # 根据复位状态设置默认值
        time_start_value = float(min_x) if st.session_state.reset_time else st.session_state.get('time_start', float(min_x))
        start_time = st.number_input(
            "起始时间",
            value=time_start_value,
            min_value=float(min_x),
            max_value=float(max_x),
            step=0.1 if isinstance(min_x, float) else 1,
            key="time_start"
        )
    with col2:
        time_end_value = float(max_x) if st.session_state.reset_time else st.session_state.get('time_end', float(max_x))
        end_time = st.number_input(
            "结束时间",
            value=time_end_value,
            min_value=float(min_x),
            max_value=float(max_x),
            step=0.1 if isinstance(min_x, float) else 1,
            key="time_end"
        )
    with col3:
        # 复位按钮：仅修改session state，不触发rerun
        if st.button("🔄 复位时间", type="secondary"):
            st.session_state.reset_time = True
            st.session_state.time_start = float(min_x)
            st.session_state.time_end = float(max_x)
        else:
            # 非点击状态时重置复位标记
            if st.session_state.reset_time:
                st.session_state.reset_time = False
    
    # 验证时间区间有效性
    if start_time > end_time:
        st.error("起始时间不能大于结束时间，请重新设置")
        return
    time_range = (start_time, end_time)
    
    # 6. 纵坐标区间选择（修复复位功能）
    st.markdown("---")
    st.subheader("纵坐标（参数值）区间筛选")
    
    # 获取当前选中参数的全局数值范围
    all_y_values = []
    for df in df_list:
        if selected_param in df.columns:
            df_time_filtered = df[(df[x_col] >= time_range[0]) & (df[x_col] <= time_range[1])]
            all_y_values.extend(df_time_filtered[selected_param].dropna().values)
    
    if not all_y_values:
        st.warning("当前时间区间内无有效参数值，无法设置纵坐标范围")
        return
    
    min_y = min(all_y_values)
    max_y = max(all_y_values)
    # 扩展10%的范围作为默认值
    default_y_min = min_y - (max_y - min_y) * 0.1 if max_y != min_y else min_y - 1
    default_y_max = max_y + (max_y - min_y) * 0.1 if max_y != min_y else max_y + 1
    st.session_state.default_y_range = (default_y_min, default_y_max)
    
    # 纵坐标区间布局：最小值 + 最大值 + 复位按钮
    col4, col5, col6 = st.columns([2, 2, 1])
    with col4:
        # 根据复位状态设置默认值
        y_min_value = float(default_y_min) if st.session_state.reset_y else st.session_state.get('y_min', float(default_y_min))
        y_min = st.number_input(
            "数值最小值",
            value=y_min_value,
            min_value=float(min_y - (max_y - min_y) * 1) if max_y != min_y else min_y - 10,
            max_value=float(max_y),
            step=0.01 if isinstance(default_y_min, float) else 1,
            key="y_min"
        )
    with col5:
        y_max_value = float(default_y_max) if st.session_state.reset_y else st.session_state.get('y_max', float(default_y_max))
        y_max = st.number_input(
            "数值最大值",
            value=y_max_value,
            min_value=float(min_y),
            max_value=float(max_y + (max_y - min_y) * 1) if max_y != min_y else max_y + 10,
            step=0.01 if isinstance(default_y_max, float) else 1,
            key="y_max"
        )
    with col6:
        # 复位按钮：仅修改session state，不触发rerun
        if st.button("🔄 复位数值", type="secondary"):
            st.session_state.reset_y = True
            st.session_state.y_min = float(default_y_min)
            st.session_state.y_max = float(default_y_max)
        else:
            # 非点击状态时重置复位标记
            if st.session_state.reset_y:
                st.session_state.reset_y = False
    
    # 验证纵坐标区间有效性
    if y_min >= y_max:
        st.error("数值最小值不能大于等于最大值，请重新设置")
        return
    y_range = (y_min, y_max)
    
    # 7. 生成并显示图表
    st.markdown("---")
    st.subheader("对比图表")
    fig = generate_time_series_plot(df_list, file_names, selected_param, x_col, time_range, y_range)
    
    # 显示图表
    st.pyplot(fig)
    
    # 8. 下载链接
    st.markdown(get_download_link(fig, selected_param, time_range, y_range), unsafe_allow_html=True)
    
    # 9. 数据预览（筛选后的数据）
    st.markdown("---")
    st.subheader("数据预览（当前时间+数值区间）")
    tab_list = st.tabs(file_names)
    for idx, tab in enumerate(tab_list):
        with tab:
            # 同时筛选时间和数值区间的数据
            df_filtered = df_list[idx][
                (df_list[idx][x_col] >= time_range[0]) & 
                (df_list[idx][x_col] <= time_range[1]) &
                (df_list[idx][selected_param] >= y_range[0]) &
                (df_list[idx][selected_param] <= y_range[1])
            ]
            st.dataframe(df_filtered, use_container_width=True)

if __name__ == "__main__":
    main()
