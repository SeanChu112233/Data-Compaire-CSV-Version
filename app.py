import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO, StringIO
import base64

# 设置matplotlib支持中文
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def load_data(uploaded_file):
    """加载CSV数据，尝试多种编码"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for encoding in encodings:
        try:
            if uploaded_file is not None:
                string_data = uploaded_file.getvalue().decode(encoding)
                return pd.read_csv(StringIO(string_data))
        except UnicodeDecodeError:
            continue
    st.error("无法解析CSV文件，请检查文件编码格式")
    return None

def generate_plot(df, scales, selected_params):
    """生成3D可交互图表"""
    # 筛选选中的参数
    filtered_params = [p for p in df.columns[1:] if p in selected_params]
    if not filtered_params:
        st.warning("请至少选择一个参数进行可视化")
        return None
        
    # 第一列为X轴数据（时间）
    x_data = df.iloc[:, 0].values
    
    # 获取参数名称（从第二列开始）并反转顺序
    reversed_params = list(reversed(filtered_params))  # 反转参数顺序
    num_params = len(reversed_params)
    
    # 创建图形和3D轴
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取颜色映射
    colors = cm.rainbow(np.linspace(0, 1, num_params))
    
    # 绘制每条数据线（使用反转后的参数顺序）
    for i, param in enumerate(reversed_params):
        # 获取Y轴位置（均匀分布）
        y_pos = i
        
        # 获取Z轴数据，并应用缩放
        z_data = df[param].values * scales[param]
        
        # 绘制3D线
        ax.plot(x_data, np.full_like(x_data, y_pos), z_data, 
                label=param, color=colors[i], linewidth=2)
    
    # 设置轴标签
    ax.set_xlabel(df.columns[0], fontsize=10)  # X轴使用第一列的名称
    ax.set_zlabel(filtered_params[0], fontsize=10)  # Z轴以第一个选中的参数为准
    
    # 设置Y轴刻度和标签（使用反转后的参数顺序）
    ax.set_yticks(range(num_params))
    ax.set_yticklabels(reversed_params, fontsize=8)  # 注意这里使用反转后的参数
    
    # 添加标题和图例
    ax.set_title('3D Waterfall compare plot', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 调整视角
    ax.view_init(elev=30, azim=290)
    
    # 优化布局
    plt.tight_layout()
    
    return fig

def generate_time_series_plots(df, selected_params, x_range):
    """生成时间序列对比图，从上到下排列各个参数，不应用缩放"""
    # 筛选选中的参数
    filtered_params = [p for p in df.columns[1:] if p in selected_params]
    if not filtered_params:
        st.warning("请至少选择一个参数进行可视化")
        return None
        
    # 获取X轴数据和范围
    x_data = df.iloc[:, 0].values
    x_label = df.columns[0]
    
    # 根据选择的x范围筛选数据
    mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
    filtered_x = x_data[mask]
    
    # 创建图形
    num_plots = len(filtered_params)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots), sharex=True)
    
    # 确保axes是数组形式（当只有一个参数时）
    if num_plots == 1:
        axes = [axes]
    
    # 为每个参数创建子图 - 不应用缩放
    for i, param in enumerate(filtered_params):
        # 获取原始Y数据（不进行缩放）
        y_data = df[param].values[mask]
        
        # 绘制曲线
        axes[i].plot(filtered_x, y_data, label=param, linewidth=2)
        axes[i].set_title(f'{param} ', fontsize=10)
        axes[i].set_ylabel(f'{param} ()', fontsize=8)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend(fontsize=8)
    
    # 设置X轴标签
    axes[-1].set_xlabel(x_label, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def get_image_download_link(fig, format='png', prefix='data'):
    """生成图表下载链接"""
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_data = buf.getvalue()
    b64 = base64.b64encode(img_data).decode()
    return f'<a href="data:image/{format};base64,{b64}" download="{prefix}_plot.{format}">下载图表</a>'

def main():
    st.title('多参数3D数据对比可视化瀑布图')
    st.write('上传CSV文件，第一列为X轴数据（如时间），第二列为Z轴基准（如速度），其他列为需要对比的参数。')
    st.write('您可以调整各参数的缩放比例以获得更好的对比效果，需要其他视角请与我联系。')
    
    # 上传文件
    uploaded_file = st.file_uploader("选择CSV文件", type="csv")
    
    if uploaded_file is not None:
        # 加载数据
        df = load_data(uploaded_file)
        
        if df is not None and not df.empty:
            # 显示数据预览
            st.subheader('数据预览')
            st.dataframe(df.head())
            
            # 检查数据列数
            if len(df.columns) < 2:
                st.error("CSV文件至少需要包含两列数据（第一列为X轴，其余为参数）")
                return
            
            # 初始化缩放比例（以第二列作为基准，缩放比例为1）
            params = df.columns[1:]
            initial_scales = {param: 1.0 for param in params}
            
            # 如果有多个参数，从会话状态加载或初始化缩放比例
            if 'scales' not in st.session_state or set(st.session_state.scales.keys()) != set(params):
                st.session_state.scales = initial_scales
            
            # 初始化参数选择状态
            if 'selected_params' not in st.session_state or set(st.session_state.selected_params) != set(params):
                st.session_state.selected_params = list(params)  # 默认全选
            
            # 参数选择
            st.subheader('参数选择')
            st.write('选择需要在图表中显示的参数')
            selected_params = st.multiselect(
                '选择参数',
                options=params,
                default=st.session_state.selected_params
            )
            st.session_state.selected_params = selected_params
            
            # 参数缩放控制
            st.subheader('参数缩放调整')
            st.write('调整各参数的缩放比例，以便更好地在图表上进行对比（第二列作为基准）')
            
            # 使用表单组织缩放控件（使用反转后的参数顺序）
            with st.form("scaling_form"):
                cols = st.columns(3)  # 使用多列布局
                reversed_params = list(reversed(params))  # 反转参数顺序
                for i, param in enumerate(reversed_params):
                    # 原第二列固定为1.0，作为基准
                    if param == params[0]:  # 无论顺序如何，始终以原第二列作为基准
                        st.session_state.scales[param] = 1.0
                        cols[i % 3].number_input(
                            f'{param} (基准)', 
                            value=1.0, 
                            min_value=0.01, 
                            max_value=100.0, 
                            step=0.1,
                            disabled=True  # 基准参数不可修改
                        )
                    else:
                        st.session_state.scales[param] = cols[i % 3].number_input(
                            param, 
                            value=st.session_state.scales[param], 
                            min_value=0.01, 
                            max_value=100.0, 
                            step=0.1
                        )
                
                # 提交按钮
                submitted = st.form_submit_button('应用缩放')
            
            # 生成并显示3D图表
            st.subheader('3D Waterfall plot')
            fig_3d = generate_plot(df, st.session_state.scales, selected_params)
            if fig_3d:
                st.pyplot(fig_3d, use_container_width=True)
                st.markdown(get_image_download_link(fig_3d, prefix='3d_waterfall'), unsafe_allow_html=True)
            
            # X轴区间选择（移到瀑布图下方）
            x_data = df.iloc[:, 0].values
            x_min, x_max = x_data.min(), x_data.max()
            
            # 初始化X轴范围
            if 'x_range' not in st.session_state:
                st.session_state.x_range = (x_min, x_max)
            
            st.subheader('X轴区间设置')
            col1, col2 = st.columns(2)
            with col1:
                x_start = st.number_input(
                    '起始值',
                    value=st.session_state.x_range[0],
                    min_value=x_min,
                    max_value=x_max,
                    step=(x_max - x_min)/100 if x_max != x_min else 1
                )
            with col2:
                x_end = st.number_input(
                    '结束值',
                    value=st.session_state.x_range[1],
                    min_value=x_min,
                    max_value=x_max,
                    step=(x_max - x_min)/100 if x_max != x_min else 1
                )
            
            # 确保起始值小于结束值
            if x_start > x_end:
                st.error("起始值不能大于结束值")
                return
                
            st.session_state.x_range = (x_start, x_end)
            
            # 生成并显示时间序列对比图（不应用缩放）
            st.subheader('参数时间序列对比图（原始数据）')
            fig_time_series = generate_time_series_plots(df, selected_params, st.session_state.x_range)
            if fig_time_series:
                st.pyplot(fig_time_series, use_container_width=True)
                st.markdown(get_image_download_link(fig_time_series, prefix='time_series'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
