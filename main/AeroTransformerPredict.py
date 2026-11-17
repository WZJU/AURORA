import torch
import numpy as np
from flowvae.app.wing.api import SimpleWingAPI, SuperWingAPI
import trimesh
import warnings
from cfdpost.wing.multi_section_wing import MultiSecWing
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance_matrix
from flowvae.post import rotate_input, intergal_output
from typing import Dict, Optional
'''
@ D:\generate_3\AeroTransformer_Project
'''

def clustcos(i: int, nn: int, a0=0.0079, a1=0.96, beta=1.0) -> float:
    """点分布函数 - 前缘和后缘加密"""
    aa = np.power((1-np.cos(a0*np.pi))/2.0, beta)
    dd = np.power((1-np.cos(a1*np.pi))/2.0, beta) - aa
    yt = i/(nn-1.0)
    a  = np.pi*(a0*(1-yt)+a1*yt)
    c  = (np.power((1-np.cos(a))/2.0,beta)-aa)/dd
    return c

def dist_clustcos(nn: int, a0=0.0079, a1=0.96, beta=1.0) -> np.ndarray:
    """生成聚类余弦分布的X坐标"""
    return np.array([clustcos(i, nn, a0, a1, beta) for i in range(nn)])

def load_stl_and_generate_cgrid_by_slices(stl_path: str, wing_params: dict = None, scale: float = None, switch_YZ: bool = True) -> tuple:
    '''
    严格按照 AeroTransformer 标准生成的结构化网格（垂向射线投射物理版）
    
    完全贯彻用户思想：
    不进行任何前置的上下表面种子划分。
    针对每一个目标目标 (X, Z) 点，直接在 STL 切片多边形上做垂向投影，
    计算出所有可能的 Y 交点。
    严格执行：最大值归上表面，最小值归下表面。完美解决反弯翼型、大扭转打架问题。
    '''
    print(f'📂 加载STL文件: {stl_path}')
    mesh = trimesh.load(stl_path)
    
    if not mesh.is_watertight:
        warnings.warn("⚠️ 输入的 STL 模型非严格闭合，切片阶段可能会产生零散空集。")
    
    vertices = mesh.vertices
    x_raw = vertices[:, 0]  
    y_raw = vertices[:, 2]  # 厚度方向
    z_raw = vertices[:, 1]  # 展长方向
    
    chord_raw = x_raw.max() - x_raw.min()
    if scale is None:
        scale = 1.0 / chord_raw
        
    CHORD_CELLS, SPAN_CELLS = 256, 128
    CHORD_NODES = CHORD_CELLS + 1 
    SPAN_NODES = SPAN_CELLS + 1   
    
    z_min, z_max = z_raw.min(), z_raw.max()
    total_span = z_max - z_min
    cut_off_ratio = 0.003  
    inner_cut_off_ratio = 0.1
    z_retained_max = z_max - (total_span * cut_off_ratio)
    z_retained_min = z_min + (total_span * inner_cut_off_ratio)
    z_stations = np.linspace(z_retained_min, z_retained_max, SPAN_NODES)
    
    half_cells = CHORD_CELLS // 2  
    # s = np.linspace(0, 1, half_cells + 1) 
    # alpha = 1.6 
    # x_dist = np.power(s, alpha)  
    x_dist = np.array([clustcos(i, half_cells + 1) for i in range(half_cells + 1)])
    
    node_grid = np.zeros((CHORD_NODES, SPAN_NODES, 3))
    
    print("⏳ 开始进行展向站位切片插值（257x129节点阵列）...")
    for j, z_val in enumerate(z_stations):
        # 1. 投影切片，获取离散的交线段
        path3d = trimesh.intersections.mesh_plane(mesh, plane_normal=[0, 1, 0], plane_origin=[0, z_val, 0])
        
        if len(path3d) == 0:
            if j == 0:
                raise ValueError("❌ 根部第一个站位切片即为空集，请检查STL模型坐标系。")
            elif j >= 2:
                dx = node_grid[:, j-1, 0] - node_grid[:, j-2, 0]
                dy = node_grid[:, j-1, 2] - node_grid[:, j-2, 2]
                node_grid[:, j, 0] = node_grid[:, j-1, 0] + dx * 0.5
                node_grid[:, j, 1] = z_val * scale
                node_grid[:, j, 2] = node_grid[:, j-1, 2] + dy * 0.5
            else:
                node_grid[:, j, :] = node_grid[:, j-1, :]
                node_grid[:, j, 1] = z_val * scale
            continue
            
        # 2. 提取当前切片的所有物理边界线段 (segments)
        # lines 的形状一般是 (N, 2, 3)，代表 N 条空间线段
        segments = path3d.reshape(-1, 2, 3)
        seg_x = segments[..., 0] # (N, 2)
        seg_y = segments[..., 2] # (N, 2)
        
        sx_min, sx_max = seg_x.min(), seg_x.max()
        local_chord = sx_max - sx_min
        
        # 提前确定前缘和后缘的绝对锚定点，用于极端边界兜底
        all_pts = path3d.reshape(-1, 3)
        sx_all, sy_all = all_pts[:, 0], all_pts[:, 2]
        sy_LE = sy_all[np.argmin(sx_all)]
        
        te_indices = np.where(sx_all > sx_max - 0.01 * local_chord)[0]
        sy_TE_low = np.min(sy_all[te_indices])
        sy_TE_up = np.max(sy_all[te_indices])

        # 生成目标 X 物理序列
        x_phys_dist = x_dist * local_chord + sx_min
        
        # 3. 🎯 核心逻辑：针对每一个目标 X 坐标，寻找穿过所有线段的 Y 交点
        def ray_cast_y(x_target):
            y_intersections = []
            # 遍历切片中的每一条微小线段
            for i_seg in range(len(seg_x)):
                x1, x2 = seg_x[i_seg, 0], seg_x[i_seg, 1]
                y1, y2 = seg_y[i_seg, 0], seg_y[i_seg, 1]
                
                # 检查目标 x_target 是否落在该线段的 X 范围内
                if (min(x1, x2) <= x_target) and (x_target <= max(x1, x2)):
                    # 避免分母为 0（垂直线段）
                    if abs(x2 - x1) < 1e-9:
                        y_intersections.append(0.5 * (y1 + y2))
                    else:
                        # 线性插值求出该线段上的 Y 值
                        t = (x_target - x1) / (x2 - x1)
                        y_curr = y1 + t * (y2 - y1)
                        y_intersections.append(y_curr)
                        
            if len(y_intersections) == 0:
                return None
            return np.array(y_intersections)

        # 4. 主动点序填充网格
        # 下表面：后缘 (s=1) -> 前缘 (s=0)
        for i in range(half_cells + 1):
            x_target_phys = x_phys_dist[half_cells - i]
            node_grid[i, j, 0] = x_target_phys * scale
            node_grid[i, j, 1] = z_val * scale
            
            if i == 0:
                node_grid[i, j, 2] = sy_TE_low * scale
            elif i == half_cells:
                node_grid[i, j, 2] = sy_LE * scale
            else:
                y_hits = ray_cast_y(x_target_phys)
                # 🛡️ 贯彻你的想法：下表面无脑取所有交点中最小的 Y
                node_grid[i, j, 2] = (np.min(y_hits) if y_hits is not None else sy_LE) * scale
                
        # 上表面：前缘后的第一个点 -> 后缘 (s=1)
        for k in range(1, half_cells + 1):
            i = half_cells + k
            x_target_phys = x_phys_dist[k]
            node_grid[i, j, 0] = x_target_phys * scale
            node_grid[i, j, 1] = z_val * scale
            
            if i == CHORD_NODES - 1:
                node_grid[i, j, 2] = sy_TE_up * scale
            else:
                y_hits = ray_cast_y(x_target_phys)
                # 🛡️ 贯彻你的想法：上表面无脑取所有交点中最大的 Y
                node_grid[i, j, 2] = (np.max(y_hits) if y_hits is not None else sy_LE) * scale

    # --- 🛡️ 邻域平滑清洗层（处理个别由于STL破洞导致的未交点） ---
    if np.any(np.isnan(node_grid)) or np.any(node_grid == 0):
        for c in range(3):
            bad_mask = np.isnan(node_grid[..., c])
            if np.any(bad_mask):
                node_grid[..., c][bad_mask] = np.interp(
                    np.flatnonzero(bad_mask), 
                    np.flatnonzero(~bad_mask), 
                    node_grid[..., c][~bad_mask]
                )
                
    print(f"✅ 节点网格(Node)生成成功。最终矩阵形状: {node_grid.shape}")
    
    cell_grid = 0.25 * (
        node_grid[:-1, :-1, :] + 
        node_grid[1:, :-1, :] + 
        node_grid[:-1, 1:, :] + 
        node_grid[1:, 1:, :]
    )
    print(f'🚀 中心点网格(Cell)计算完毕！最终矩阵形状: {cell_grid.shape}')
    
    final_chord = (x_raw.max() - x_raw.min()) * scale
    final_span = (z_raw.max() - z_raw.min()) * scale
    
    if switch_YZ:
        node_grid[:, :, [1, 2]] = node_grid[:, :, [2, 1]] # 交换 Z 和 Y 轴.GGS输出的Y轴是翼展方向,而模型输入的Y轴是垂直方向
        cell_grid[:, :, [1, 2]] = cell_grid[:, :, [2, 1]] # 交换 Z 和 Y 轴.GGS输出的Y轴是翼展方向,而模型输入的Y轴是垂直方向
    node_grid = node_grid.transpose(2, 1, 0)
    cell_grid = cell_grid.transpose(2, 1, 0)
    return cell_grid, node_grid, wing_params, (final_chord, final_span)

class AeroTransformer():
    '''
    加载Aerotranformer模型.对某一几何,使用build_grid方法构建网格,然后使用predict_coeffs方法预测系数,也可以使用predict_flow_field方法预测流场,然后使用plot_flow_field方法可视化流场.
    '''
    def __init__(self, weights_path='D:/generate_3/AeroTransformer_Project/weights', model_version='default'):
        self.model = self._load_model(weights_path, model_version)
        self.device = getattr(self.model, 'device', 'cuda')
        self.model.setAP(None)
        self.node_grid = None
        self.cell_grid = None
    
    def _load_model(self, weights_path='D:/generate_3/AeroTransformer_Project/weights', model_version='default'):
        '''加载预训练AeroTransformer模型'''
        print(f'🚀 加载AeroTransformer模型...')
        model = SuperWingAPI(saves_folder=weights_path, model_version=model_version)
        # print(f'✅ 模型加载成功！包含3个子模型：2D翼型模型 + SLD模型 + 3D全机模型')
        return model
    
    def build_grid(self, file_path: str, root_chord: float, area: float, mac: float):
        # 设置模型参考参数
        class DummyAeroProblem:
            def __init__(self):
                self.areaRef = area/(root_chord**2)   # 参考面积 (S_ref)
                self.chordRef = mac/root_chord  # 参考弦长 (c_ref)
                self.xRef = 0.0      
                self.yRef = 0.0
        ap_object = DummyAeroProblem()
        self.model.setAP(ap_object)
        '''构建STL文件的网格'''
        cell_grid, node_grid, wing_params, (chord, half_span) = load_stl_and_generate_cgrid_by_slices(file_path, scale=1/root_chord, switch_YZ=True)
        # node_grid[:, :, [1, 2]] = node_grid[:, :, [2, 1]] # 交换 Z 和 Y 轴.GGS输出的Y轴是翼展方向,而模型输入的Y轴是垂直方向
        self.node_grid = node_grid
        self.cell_grid = cell_grid

    def predict_coeffs(self, aoa_array: np.ndarray, mach_array: np.ndarray):
        if self.node_grid is None or self.cell_grid is None:
            raise ValueError('请先使用build_grid方法构建网格')
        '''预测系数'''
        assert aoa_array.shape == mach_array.shape
        cond_tensor = torch.tensor(np.column_stack((aoa_array, mach_array)), dtype=torch.float32)
        assert cond_tensor.shape[1] == 2

        geom = self.node_grid # (CHORD_POINTS, SPAN_POINTS, 3)变为(3, SPAN_POINTS, CHORD_POINTS)

        geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).repeat(cond_tensor.shape[0], 1, 1, 1).to(self.device)
        # print(geom_tensor.shape, cond_tensor.shape)
        forces = self.model.predict(inp=geom_tensor, cnd=cond_tensor.to(self.device))
        forces = forces.cpu().detach().numpy()
        return forces
    
    def predict_flow_field(self, aoa: float, mach: float):
        cond_tensor = torch.tensor([[aoa, mach]], dtype=torch.float32)
        auxs = cond_tensor.to(self.device)

        geom = self.node_grid   # -> (3, 129, 257)
        wg = MultiSecWing(aoa=aoa, iscentric=True)
        wg.read_formatted_surface(geometry=geom, isinitg=True)

        origeom, geom = wg.get_all_geometries()
        inputs = torch.from_numpy(geom).float().to(self.device).unsqueeze(0)
        flowfield = self.model._predict(inputs, auxs).cpu().detach().numpy()
        return flowfield, origeom, geom
    
    def plot_flow_field(self, flowfield: np.ndarray, origeom: np.ndarray, channel:str='pressure'):
        '''可视化流场'''
        # 现在坐标顺序为 (x, y, z) = (弦向, 垂直, 展向)
        X = origeom[0, :, :]   # 形状: (257, 129)
        Y = origeom[1, :, :]   # 形状: (257, 129)
        Z = origeom[2, :, :]   # 形状: (257, 129)

        # 选择要染色的通道
        if channel == 'pressure':
            color_channel = 0  # 选择第0通道（压力系数）
        elif channel == 'friction':
            color_channel = 1  # 选择第1通道（摩擦系数）
        else:
            raise ValueError(f'channel must be "pressure" or "friction", got {channel}')

        # 检查 output 形状
        # print(f"flowfield shape: {flowfield.shape}")

        # 根据 output 形状进行适配
        if flowfield.ndim == 4:
            # flowfield 形状: (batch, channels, span, chord)
            color_data = flowfield[0, color_channel, :, :]  # 形状: (128, 256)
        elif flowfield.ndim == 3:
            if flowfield.shape[0] <= 10:
                color_data = flowfield[color_channel, :, :]
            else:
                color_data = flowfield[:, :, color_channel]

        # print(f"color_data shape: {color_data.shape}")
        # print(f"X shape: {X.shape}")


        # print(f"color_data_transposed shape: {color_data.shape}")

        # 创建3D可视化
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 归一化颜色数据
        color_normalized = (color_data - color_data.min()) / (color_data.max() - color_data.min() + 1e-10)

        # 使用彩虹色系染色
        surf = ax.plot_surface(X, Y, Z,
                            rstride=1, cstride=1,
                            facecolors=plt.cm.rainbow(color_normalized),  # 彩虹色系
                            edgecolor='k',
                            linewidth=0.1,
                            antialiased=True,
                            alpha=1.0)

        # ============ 保持原始尺度 ============
        # 计算各轴的实际范围
        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        z_range = Z.max() - Z.min()
        
        # 设置坐标轴比例为实际数据比例
        ax.set_box_aspect([x_range, y_range, z_range])
        
        # 可选：设置坐标轴范围以显示完整数据
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(Z.min(), Z.max())

        # ============ 设置相机位置：从 y 轴正方向向 y 轴负方向看 ============
        ax.view_init(elev=0, azim=90)  # elev=仰角, azim=方位角


        # 创建颜色条 - 添加 ax=ax 参数
        norm = plt.Normalize(color_data.min(), color_data.max())
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label=f'通道 {color_channel} 值')

        ax.set_xlabel('X (弦向)')
        ax.set_ylabel('Y (垂直)')
        ax.set_zlabel('Z (展向)')
        ax.set_title(f'C 型网格 3D 可视化 - 通道 {color_channel} 染色')

        plt.show()
    
    # def integral_coeffs(self, inp: torch.Tensor, outputs: torch.Tensor, cnd: np.ndarray):
    #     ### 这个函数有点问题，先放在这不管了，等有时间再看
    #     '''计算积分系数'''
    #     forces = intergal_output(inp, outputs, cnd[:, 0], 
    #                              s=self.model.ap.areaRef, c=self.model.ap.chordRef, xref=self.model.ap.xRef, yref=self.model.ap.yRef)
    #     return forces

def integrate_wing_aerodynamics(inp, flow_field, s_ref, c_ref, x_ref, y_ref, alpha_deg, include_friction=True):
    """
    通过直接物理方向分解，积分计算半侧机翼气动力系数
    
    坐标轴定义:
    x 轴: 沿弦长指向后 | y 轴: 指向上 | z 轴: 沿翼展指向外
    """
    device = inp.device
    
    # 1. 提取网格顶点
    grid = inp[0]  # [3, 129, 257]
    
    p00 = grid[:, :-1, :-1]  # [3, 128, 256]
    p10 = grid[:, 1:,  :-1]  # i+1 方向 (展向)
    p11 = grid[:, 1:,  1:]
    p01 = grid[:, :-1, 1:]   # j+1 方向 (弦向展开)
    
    # 计算微元中心坐标 (用于力矩计算)
    x_elem = (p00 + p10 + p11 + p01) / 4.0  # [3, 128, 256]
    
    # 2. 几何计算：面积与单位法向量 n (采用空间对角线法防扭曲)
    diag1 = p11 - p00
    diag2 = p10 - p01
    
    # 叉乘计算面积向量
    # 注意：根据你的测试，去掉负号算出来的 CD 如果正常，说明此时 dA_vector 为正确的内法线方向
    dA_vector = 0.5 * torch.cross(diag1, diag2, dim=0)  # [3, 128, 256]
    dA_mag = torch.norm(dA_vector, dim=0, keepdim=True)  # 微元面积 A_i [1, 128, 256]
    n = dA_vector / (dA_mag + 1e-8)                      # 单位内法向量 [3, 128, 256]
    
    # 3. 提取流场压力系数
    cp = flow_field[0, 0].unsqueeze(0)        # [1, 128, 256]
    
    # 4. 计算纯压力在 xyz 三个方向产生的微元力 (Cp * n * A_i)
    F_elem_x = cp * n[0:1] * dA_mag
    F_elem_y = cp * n[1:2] * dA_mag
    F_elem_z = cp * n[2:3] * dA_mag
    
    # 5. 计算摩擦力 (对应公式中的纯切向项: Cf - Cf·n)
    if include_friction:
        cf_tau = flow_field[0, 1].unsqueeze(0)    # [1, 128, 256] 弦向摩擦
        cf_z = flow_field[0, 2].unsqueeze(0)      # [1, 128, 256] 展向摩擦
        
        # 提取网格自带的物理切方向边
        v_surf_tau = p01 - p00  # 沿网格弦向展开的天然切向向量
        v_surf_z   = p10 - p00  # 沿网格展向展开的天然切向向量
        
        # 单位化这两个切方向向量
        t_tau = v_surf_tau / (torch.norm(v_surf_tau, dim=0, keepdim=True) + 1e-8)
        t_z   = v_surf_z / (torch.norm(v_surf_z, dim=0, keepdim=True) + 1e-8)
        
        # 摩擦力直接沿这两个方向作用，并乘以面积 A_i
        # 对应公式里的: (Cf - Cf·n) * A_i
        F_fric_x = (cf_tau * t_tau[0:1] + cf_z * t_z[0:1]) * dA_mag
        F_fric_y = (cf_tau * t_tau[1:2] + cf_z * t_z[1:2]) * dA_mag
        F_fric_z = (cf_tau * t_tau[2:3] + cf_z * t_z[2:3]) * dA_mag
        
        # 叠加到总微元力中
        F_elem_x += F_fric_x
        F_elem_y += F_fric_y
        F_elem_z += F_fric_z
    
    # 6. 计算微元对参考点的俯仰力矩 (绕 z 轴)
    r_x = x_elem[0:1] - x_ref
    r_y = x_elem[1:2] - y_ref
    M_z_elem = r_x * F_elem_y - r_y * F_elem_x
    
    # 7. 全局求和积分
    F_body_x = torch.sum(F_elem_x)
    F_body_y = torch.sum(F_elem_y)
    M_z_total = torch.sum(M_z_elem)
    
    # 8. 坐标变换：机体轴系 (Body) -> 风轴系 (Wind)
    alpha_rad = np.radians(alpha_deg)
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    
    # 转换逻辑：x轴向后，y轴向上
    D =  F_body_x * cos_a + F_body_y * sin_a
    L = -F_body_x * sin_a + F_body_y * cos_a
    
    # 9. 无量纲化
    CL = L / s_ref
    CD = D / s_ref
    Cm = M_z_total / (s_ref * c_ref)
    
    return {
        "CL": CL.item(),
        "CD": CD.item(),
        "CM": Cm.item()
    }
'''
def integrate_wing_aerodynamics(inp, flow_field, s_ref, c_ref, x_ref, y_ref, alpha_deg, include_friction=True):
    """
    通过直接物理方向分解，积分计算半侧机翼气动力系数
    
    参数:
    ----------
    inp : torch.Tensor [1, 3, 129, 257] -> 机翼表面网格节点坐标
    flow_field : torch.Tensor [1, 3, 128, 256] -> 流场数据 (Cp, cf_tau, cf_z)
    s_ref, c_ref : float -> 参考面积, 参考弦长
    x_ref, y_ref : float -> 力矩参考点位置
    alpha_deg : float -> 攻角 (单位: 度)
    include_friction : bool -> 是否计算摩擦阻力开关 (默认为 True)
    """
    device = inp.device
    
    # 1. 提取网格顶点
    grid = inp[0]  # [3, 129, 257]
    
    p00 = grid[:, :-1, :-1]  # [3, 128, 256]
    p10 = grid[:, 1:,  :-1]
    p11 = grid[:, 1:,  1:]
    p01 = grid[:, :-1, 1:]
    
    # 计算微元中心坐标
    x_elem = (p00 + p10 + p11 + p01) / 4.0  # [3, 128, 256]
    
    # 2. 几何计算：精确的面积与单位内法向量 n (对角线法)
    diag1 = p11 - p00
    diag2 = p10 - p01
    
    # 取负号调整为内法线 (压向机翼内部)
    dA_vector = 0.5 * torch.cross(diag1, diag2, dim=0)  # [3, 128, 256]
    dA_mag = torch.norm(dA_vector, dim=0, keepdim=True)  # 微元面积 A_i [1, 128, 256]
    n = dA_vector / (dA_mag + 1e-8)                      # 单位内法向量 [3, 128, 256]
    
    # 3. 提取流场分布系数
    cp = flow_field[0, 0].unsqueeze(0)        # [1, 128, 256]
    
    # 4. 初始化绝对坐标系 (x, y, z) 下的微元力
    # 4.1 压力在 xyz 三个方向产生的微元力 (基本气动力，始终计算)
    F_elem_x = cp * n[0:1] * dA_mag
    F_elem_y = cp * n[1:2] * dA_mag
    F_elem_z = cp * n[2:3] * dA_mag
    
    # 5. 根据开关条件计算并叠加摩擦力
    if include_friction:
        cf_tau = flow_field[0, 1].unsqueeze(0)    # [1, 128, 256]
        cf_z = flow_field[0, 2].unsqueeze(0)      # [1, 128, 256]
        
        # 定义绝对坐标轴
        x_axis = torch.zeros_like(n); x_axis[0, :, :] = 1.0
        z_axis = torch.zeros_like(n); z_axis[2, :, :] = 1.0
        
        # 将绝对轴正交投影到微元切平面，构建严格的切方向
        x_dot_n = torch.sum(x_axis * n, dim=0, keepdim=True)
        t_tau_raw = x_axis - x_dot_n * n
        t_tau = t_tau_raw / (torch.norm(t_tau_raw, dim=0, keepdim=True) + 1e-8)
        
        z_dot_n = torch.sum(z_axis * n, dim=0, keepdim=True)
        t_z_raw = z_axis - z_dot_n * n
        t_z = t_z_raw / (torch.norm(t_z_raw, dim=0, keepdim=True) + 1e-8)
        
        # 叠加摩擦力分量
        F_elem_x += (cf_tau * t_tau[0:1] + cf_z * t_z[0:1]) * dA_mag
        F_elem_y += (cf_tau * t_tau[1:2] + cf_z * t_z[1:2]) * dA_mag
        F_elem_z += (cf_tau * t_tau[2:3] + cf_z * t_z[2:3]) * dA_mag
    
    # 6. 计算微元对参考点的俯仰力矩 (绕 z 轴)
    r_x = x_elem[0:1] - x_ref
    r_y = x_elem[1:2] - y_ref
    M_z_elem = r_x * F_elem_y - r_y * F_elem_x
    
    # 7. 全局积分 (求和)
    F_body_x = torch.sum(F_elem_x)
    F_body_y = torch.sum(F_elem_y)
    M_z_total = torch.sum(M_z_elem)
    
    # 8. 坐标变换：机体轴系 (Body) -> 风轴系 (Wind)
    alpha_rad = np.radians(alpha_deg)
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    
    # x轴向后，y轴向上
    D =  F_body_x * cos_a + F_body_y * sin_a
    L = -F_body_x * sin_a + F_body_y * cos_a
    
    # 9. 无量纲化
    CL = L / s_ref
    CD = D / s_ref
    Cm = M_z_total / (s_ref * c_ref)
    
    return {
        "CL": CL.item(),
        "CD": CD.item(),
        "CM": Cm.item()
    }
'''

'''
def integrate_half_wing_aero_coeffs(
    inp: torch.Tensor,              # [B, 3, 129, 257]
    flow_field: torch.Tensor,       # [B, 3, 128, 256]
    S_ref: float,
    alpha_deg: float,
    ref_point: torch.Tensor,        # [3] or [B, 3], reference point for moments
    c_ref: Optional[float] = None,
    b_ref: Optional[float] = None,
    mirror_to_full: bool = True,
    cf_is_force_on_body: bool = True,
    auto_flip_normal: bool = True,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    from chatGPT 5.4
    对半翼表面气动力分布积分，得到力/力矩系数。

    坐标系:
        x: 沿弦向指向后
        y: 向上
        z: 沿翼展向外（左侧）

    inp:
        [B, 3, Nz+1, Ns+1] = [B, 3, 129, 257]
        channel 0/1/2 -> x/y/z
        dim=2: spanwise, root -> tip
        dim=3: surface wrap, TE lower -> LE -> upper -> TE

    flow_field:
        [B, 3, Nz, Ns] = [B, 3, 128, 256]
        channel 0 -> Cp
        channel 1 -> cf_tau   (局部x-y截面切向)
        channel 2 -> cf_z     (翼展方向)

    参数
    ----
    S_ref: 参考面积
    alpha_deg: 攻角，单位度
    ref_point: 力矩参考点 [3] 或 [B,3]
    c_ref: 弦长参考长度，用于俯仰力矩无量纲化
    b_ref: 翼展参考长度，用于滚转/偏航力矩无量纲化
    mirror_to_full: 若为True，默认将半翼积分结果乘2作为整翼
    cf_is_force_on_body: 若True，cf方向视为“流体对机翼”的方向；若False则摩擦项取负
    auto_flip_normal: 自动检查并修正法向朝外
    eps: 数值稳定项

    返回
    ----
    dict:
        CF_xyz: [B,3]   机体系/当前坐标轴下的力系数 (Cx, Cy, Cz)
        CM_xyz: [B,3]   机体系下绕参考点力矩系数 (Cl_roll, Cm_pitch_dimless?, Cn_yaw_dimless?)
        CL:     [B]
        CD:     [B]
        CY_wind:[B]     侧力系数（风轴中）
        F_xyz_coeff_pressure: [B,3]
        F_xyz_coeff_friction: [B,3]
        area_total: [B]  实际积分的半翼/整翼面积（若mirror则为乘2后）
    """

    assert inp.ndim == 4 and flow_field.ndim == 4
    B, Cg, Nz1, Ns1 = inp.shape
    Bf, Cf, Nz, Ns = flow_field.shape
    assert B == Bf
    assert Cg == 3 and Cf == 3
    assert Nz1 == Nz + 1 and Ns1 == Ns + 1

    device = inp.device
    dtype = inp.dtype

    # ref_point shape -> [B, 3]
    if ref_point.ndim == 1:
        ref_point = ref_point.to(device=device, dtype=dtype).unsqueeze(0).expand(B, 3)
    else:
        ref_point = ref_point.to(device=device, dtype=dtype)
        assert ref_point.shape == (B, 3)

    # -------------------------
    # 1) 取四角点，构建面元
    # -------------------------
    # p00 ---- p01
    #  |       |
    # p10 ---- p11
    #
    # dim=2 spanwise, dim=3 circumferential
    p00 = inp[:, :, :-1, :-1]   # [B,3,Nz,Ns]
    p10 = inp[:, :, 1:, :-1]
    p01 = inp[:, :, :-1, 1:]
    p11 = inp[:, :, 1:, 1:]

    # 面元中心
    pc = 0.25 * (p00 + p10 + p01 + p11)   # [B,3,Nz,Ns]

    # -------------------------
    # 2) 用双三角形法计算面积矢量
    # -------------------------
    # triangle 1: p00, p10, p11
    v1 = p10 - p00
    v2 = p11 - p00
    A1 = 0.5 * torch.cross(v1, v2, dim=1)  # [B,3,Nz,Ns]

    # triangle 2: p00, p11, p01
    v3 = p11 - p00
    v4 = p01 - p00
    A2 = 0.5 * torch.cross(v3, v4, dim=1)

    Avec = A1 + A2  # 四边形面积矢量
    dS = torch.linalg.norm(Avec, dim=1, keepdim=True).clamp_min(eps)  # [B,1,Nz,Ns]
    n_hat = Avec / dS  # [B,3,Nz,Ns]

    # -------------------------
    # 3) 自动修正法向为“朝外”
    # -------------------------
    if auto_flip_normal:
        # 对每个span站，估计截面中心（对周向平均）
        # center: [B,3,Nz]
        section_center = pc.mean(dim=3)  # [B,3,Nz]
        radial = pc - section_center.unsqueeze(-1)  # [B,3,Nz,Ns]

        # 若 n_hat 与 radial 点积为负，说明法向朝内，应翻转
        dot_nr = (n_hat * radial).sum(dim=1, keepdim=True)  # [B,1,Nz,Ns]
        flip = torch.where(dot_nr < 0.0, -1.0, 1.0).to(dtype=dtype, device=device)
        n_hat = n_hat * flip
        Avec = Avec * flip

    # -------------------------
    # 4) 构造局部切向基
    # -------------------------
    # 周向方向切向量：用两侧差分更平滑，这里简化用 p01-p00 + p11-p10 平均
    t_s = 0.5 * ((p01 - p00) + (p11 - p10))  # [B,3,Nz,Ns]

    # 取其在 x-y 平面投影，作为 tau 方向
    t_tau = t_s.clone()
    t_tau[:, 2:3, :, :] = 0.0
    t_tau_norm = torch.linalg.norm(t_tau, dim=1, keepdim=True).clamp_min(eps)
    t_tau_hat = t_tau / t_tau_norm

    # 全局 z 方向单位向量
    e_z = torch.zeros((B, 3, 1, 1), device=device, dtype=dtype)
    e_z[:, 2, 0, 0] = 1.0

    # -------------------------
    # 5) 读取气动系数分布
    # -------------------------
    Cp = flow_field[:, 0:1, :, :]       # [B,1,Nz,Ns]
    cf_tau = flow_field[:, 1:2, :, :]
    cf_z = flow_field[:, 2:3, :, :]

    # -------------------------
    # 6) 压力和摩擦的面元力系数贡献
    #    dC = dF / (q_inf * S_ref)
    # -------------------------
    dC_pressure = -Cp * Avec / S_ref  # [B,3,Nz,Ns]

    friction_sign = 1.0 if cf_is_force_on_body else -1.0
    cf_vec = cf_tau * t_tau_hat + cf_z * e_z
    dC_friction = friction_sign * cf_vec * dS / S_ref

    dC_total = dC_pressure + dC_friction

    # -------------------------
    # 7) 积分总力系数
    # -------------------------
    CF_p = dC_pressure.sum(dim=(2, 3))   # [B,3]
    CF_f = dC_friction.sum(dim=(2, 3))
    CF = dC_total.sum(dim=(2, 3))

    # 同时统计面积
    area_total = dS.sum(dim=(2, 3)).squeeze(1)  # [B]

    # -------------------------
    # 8) 力矩系数
    #    dM = r x dF
    #    => dCM = (r x dC_force) / L_ref
    #
    # 对不同轴常用不同参考长度：
    # roll  about x -> b_ref
    # pitch about y -> c_ref
    # yaw   about z -> b_ref
    # -------------------------
    r = pc - ref_point.view(B, 3, 1, 1)  # [B,3,Nz,Ns]
    dCM_raw = torch.cross(r, dC_total, dim=1)  # [B,3,Nz,Ns], still needs divide by ref length
    M_coeff_raw = dCM_raw.sum(dim=(2, 3))      # [B,3]

    CM = M_coeff_raw.clone()

    # 分轴无量纲化
    if b_ref is not None:
        CM[:, 0] = CM[:, 0] / b_ref   # roll
        CM[:, 2] = CM[:, 2] / b_ref   # yaw
    if c_ref is not None:
        CM[:, 1] = CM[:, 1] / c_ref   # pitch

    # 若未提供，则保留“除以S_ref后的长度量纲形式”，不完全无量纲
    # 建议用户提供 c_ref / b_ref

    # -------------------------
    # 9) 半翼镜像到整翼
    # -------------------------
    if mirror_to_full:
        CF_p = 2.0 * CF_p
        CF_f = 2.0 * CF_f
        CF = 2.0 * CF
        area_total = 2.0 * area_total

        # 力矩镜像要注意对称性
        # 当前半翼是 z>0（左翼）还是某侧，未明确说明。
        # 在“左右对称且流场对称”的假设下：
        # - Fx, Fy 加倍
        # - Fz 相互抵消 -> 总Fz = 0
        # 对于力矩:
        # - Roll Mx 抵消 -> 0
        # - Pitch My 加倍
        # - Yaw Mz 抵消 -> 0
        #
        # 因此若 mirror_to_full 代表“对称整翼”，应按对称关系处理，而不是简单乘2。
        CF[:, 2] = 0.0
        CF_p[:, 2] = 0.0
        CF_f[:, 2] = 0.0

        if b_ref is not None or c_ref is not None:
            CM[:, 0] = 0.0
            CM[:, 1] = 2.0 * (M_coeff_raw[:, 1] / c_ref if c_ref is not None else M_coeff_raw[:, 1])
            CM[:, 2] = 0.0
        else:
            # 未给参考长度时，只有俯仰项按对称加倍，其余置零
            CM[:, 0] = 0.0
            CM[:, 1] = 2.0 * M_coeff_raw[:, 1]
            CM[:, 2] = 0.0

    # -------------------------
    # 10) 转换到升阻侧力
    # -------------------------
    alpha = torch.tensor(alpha_deg, device=device, dtype=dtype) * torch.pi / 180.0
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    # Drag positive along +x when alpha=0
    e_D = torch.stack([ca, sa, torch.tensor(0.0, device=device, dtype=dtype)]).view(1, 3)
    e_L = torch.stack([-sa, ca, torch.tensor(0.0, device=device, dtype=dtype)]).view(1, 3)
    e_Y = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)

    CD = (CF * e_D).sum(dim=1)
    CL = (CF * e_L).sum(dim=1)
    CY_wind = (CF * e_Y).sum(dim=1)

    return {
        "CF_xyz": CF,                      # [B,3]
        "CM_xyz": CM,                      # [B,3]
        "CL": CL,                          # [B]
        "CD": CD,                          # [B]
        "CY_wind": CY_wind,                # [B]
        "F_xyz_coeff_pressure": CF_p,      # [B,3]
        "F_xyz_coeff_friction": CF_f,      # [B,3]
        "area_total": area_total,          # [B]
    }
'''

#================ 气动力误差计算辅助函数 ==================
def distance_cal(Target: np.ndarray, Polar: np.ndarray):
    d1, d2, pairs = directed_hausdorff(Target, Polar)#第一个值是第一个点集到第二个点集的单向豪斯多夫距离；第二个值是第二个点集到第一个点集的单向豪斯多夫距离；第三个值是两个点集中对应点对的索引。
    return d1#根据需求，返回第一个值

def average_nearest_neighbor_distance(Target: np.ndarray, Polar: np.ndarray):
    if not np.any(Polar):
        return 10
    dist_matrix = distance_matrix(Target, Polar)
    nearest_distances_1_to_2 = [min(row) for row in dist_matrix]
    avg_dist_1_to_2 = sum(nearest_distances_1_to_2) / len(Target)
    return avg_dist_1_to_2

def find_nearest_two_numbers(lst: np.ndarray, target: float):
    nearest1 = None
    nearest2 = None
    min_diff1 = float('inf')
    min_diff2 = float('inf')
    pos1 = None
    pos2 = None
    for index, num in enumerate(lst):
        diff = abs(num - target)
        if num < target:
            if diff < min_diff1:
                min_diff2 = min_diff1
                nearest2 = nearest1
                pos2 = pos1
                min_diff1 = diff
                nearest1 = num
                pos1 = index
            elif diff < min_diff2 and num!=nearest1:
                min_diff2 = diff
                nearest2 = num
                pos2 = index
        else:
            if diff < min_diff2:
                min_diff2 = diff
                nearest2 = num
                pos2 = index
            elif diff < min_diff1 and num!=nearest2:
                min_diff1 = diff
                nearest1 = num
                pos1 = index
    return nearest1, pos1, nearest2, pos2

if __name__ == '__main__':
    aero_transformer_predict = AeroTransformer()
    aero_transformer_predict.build_grid(file_path='D:/generate_3/AeroTransformer_Project/test.stl', root_chord=6, area=58, mac=3.7)
    flowfield, origeom, geom = aero_transformer_predict.predict_flow_field(aoa=12.0, mach=0.8)
    aero_transformer_predict.plot_flow_field(flowfield, origeom, channel='pressure')





    


    
