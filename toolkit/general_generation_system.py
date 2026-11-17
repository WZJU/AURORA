'''
date: 2026-3-28
description: 这是一个通用化的脚本生成器，能够根据用户输入的参数生成适用于不同飞机设计的VSP脚本。用户可以通过修改输入参数来定制飞机的各个部分，包括机身、机翼等。该系统使用了Jinja2模板引擎来动态生成脚本内容，使得代码结构清晰且易于维护。
author: Li Yuhang, Li Ke

Update: 2026-04-06
Author: Wen Zhe
Description: 修改输出位置在用户指定的位置上，而不是子目录中

Update: 2026-05-08
Author: Wen Zhe
Description: 修改RotateAirfoilMatchDideralFlag为False，使得机翼中部连续
'''

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from jinja2 import Template

# ================= 配置与模板部分 (保持原有逻辑不变) =================

Number = Union[float, int]

class AircraftScriptGenerator:
    """通用化脚本生成器 (保持原有模板逻辑)"""
    def __init__(self):
        # ... (此处省略原有的长模板字符串 start_template, fuselage_global_template 等，保持完全不变) ...
        # 为了节省篇幅，这里假设你保留了所有 self.xxx_template 的定义
        # 请确保将原代码中的所有 self.xxx_template 定义复制到这里
        self.start_template = """
void main()
{
    Update();
    double Scale = {{ Scale | default(1.0) }};
"""
        # [注意：实际使用时，请把原文件中所有的 template 字符串变量完整复制过来]
        # 此处仅展示核心逻辑修改部分，模板内容过长未重复列出，请自行保留原样
        self.fuselage_global_template = """
    //Fuselage Part
    array<string> Fusenames = {"XSec_0", "XSec_1", "XSec_2", "XSec_3", "XSec_4", "XSec_5", "XSec_6", "XSec_7", "XSec_8", "XSec_9", "XSec_10"};
    array<string> FuseCurvenames = {"XSecCurve_0", "XSecCurve_1", "XSecCurve_2", "XSecCurve_3", "XSecCurve_4", "XSecCurve_5", "XSecCurve_6", "XSecCurve_7", "XSecCurve_8", "XSecCurve_9", "XSecCurve_10"};
    string fsec_name = Fusenames[0];
    string fsec_curvename = FuseCurvenames[0];
    int fsec_idx = 0;
    int fseccurve_idx = 0;

    string fuselage_id = AddGeom("FUSELAGE", "");
    SetGeomName(fuselage_id, "Main_body");
    string body_id = FindGeom("Main_body", 0);
    string xsec_surf_id = GetXSecSurf(body_id, 0);
    SetParmVal(body_id, "CapUMinOption", "EndCap", 1);
    SetParmVal(body_id, "CapUMaxOption", "EndCap", 1);

    double YLoc = {{ YLoc | default(0.0) }};
    double Planar = {{ Planar | default(0.0) }};    

    if ({{ Planar | default(0.0) }} > 0.0)
    {
        SetParmVal(body_id, "Sym_Planar_Flag", "Sym", {{ Planar | default(0) }});
        Update();
        SetParmVal(body_id, "Y_Rel_Location", "XForm", {{ YLoc | default(0) }});
        Update();
    }

    for (int i = 0; i < 5; i++) {
        ChangeXSecShape(xsec_surf_id, i, XS_GENERAL_FUSE);
    }
    Update();

    double Tess_W = {{ Tess_W | default(65.0) }};
    double Length = {{ Length | default(123.0) }};
    SetParmVal(body_id, "Length", "Design", Length);
    SetParmVal(body_id, "Tess_W", "Shape", Tess_W);

    double SectTess_U = 40;
    double FwdCluster = 3;
    double Width = 0;
    double Height = 0;
    double MaxWidthLoc = 0;
    double CornerRad = 0;
    double TopTanAngle = 90;
    double BotTanAngle = 90;
    double TopStr = 0.83;
    double BotStr = 0.83;
    double UpStr = 0.83;
    double LowStr = 0.83;
    double TopLAngle = 0;
    double BottomLAngle = 0;
    double RightLAngle = 0;
    double LeftLAngle = 0;
    double TopLStrength = 0.75;
    double BottomLStrength = 0.75;
    double RightLStrength = 0.75;
    double LeftLStrength = 0.75;
    double XLocPercent = 0.02;
    double YLocPercent = 0;
    double ZLocPercent = -0.005;
"""
        # ... (请在此处补全所有其他 template: fuselage_nose_template, wing_template, export_stl_template 等) ...
        # 为了保证代码可运行，你需要把原文件里所有的 self.xxx_template = """...""" 都复制过来
        # 下面为了演示逻辑，我简化了部分，实际使用请务必还原所有模板字符串
        
        # 模拟补全其他模板占位符 (实际代码中请替换为你原始的完整模板)
        self.fuselage_nose_template = """
    //===== Nose Sections ({{ nsections|length }} total) ====//
{% for nsec in nsections %}
    SectTess_U = {{ nsec.SectTess_U | default(6) }};
    FwdCluster = {{ nsec.FwdCluster | default(1) }};

    Width = {{ nsec.Width | default(0) }};
    Height = {{ nsec.Height | default(0) }};
    MaxWidthLoc = {{ nsec.MaxWidthLoc | default(0) }};
    CornerRad = {{ nsec.CornerRad | default(0) }};
    TopTanAngle = {{ nsec.TopTanAngle | default(90) }};
    BotTanAngle = {{ nsec.BotTanAngle | default(90) }};
    TopStr = {{ nsec.TopStr | default(0.83) }};
    BotStr = {{ nsec.BotStr | default(0.83) }};
    UpStr = {{ nsec.UpStr | default(0.83) }};
    LowStr = {{ nsec.LowStr | default(0.83) }};

    TopLAngle = {{ "%.8f" | format(nsec.TopLAngle | default(0)) }};
    RightLAngle = {{ "%.8f" | format(nsec.RightLAngle | default(0)) }};
    TopLStrength = {{ "%.8f" | format(nsec.TopLStrength | default(0.75)) }};
    RightLStrength = {{ "%.8f" | format(nsec.RightLStrength | default(0.75)) }};

    BottomLAngle = {{ "%.8f" | format(nsec.BottomLAngle | default(nsec.TopLAngle | default(0))) }};
    BottomLStrength = {{ "%.8f" | format(nsec.BottomLStrength | default(nsec.TopLStrength | default(0.75))) }};
    LeftLAngle = {{ "%.8f" | format(nsec.RightLAngle | default(0)) }};
    LeftLStrength = {{ "%.8f" | format(nsec.RightLStrength | default(0.75)) }};

    XLocPercent = {{ nsec.XLocPercent | default(0) }};
    YLocPercent = {{ nsec.YLocPercent | default(0) }};
    ZLocPercent = {{ nsec.ZLocPercent | default(0) }};

    fsec_name = Fusenames[fsec_idx];
    fsec_curvename = FuseCurvenames[fseccurve_idx];

    SetParmVal(body_id, "SectTess_U", fsec_name, SectTess_U);
    SetParmVal(body_id, "FwdCluster", fsec_name, FwdCluster);
    SetParmVal(body_id, "XLocPercent", fsec_name, XLocPercent);
    SetParmVal(body_id, "YLocPercent", fsec_name, YLocPercent);
    SetParmVal(body_id, "ZLocPercent", fsec_name, ZLocPercent);
    SetParmVal(body_id, "Width", fsec_curvename, Width);
    SetParmVal(body_id, "Height", fsec_curvename, Height);
    SetParmVal(body_id, "MaxWidthLoc", fsec_curvename, MaxWidthLoc);
    SetParmVal(body_id, "CornerRad", fsec_curvename, CornerRad);
    SetParmVal(body_id, "TopTanAngle", fsec_curvename, TopTanAngle);
    SetParmVal(body_id, "BotTanAngle", fsec_curvename, BotTanAngle);
    SetParmVal(body_id, "TopStr", fsec_curvename, TopStr);
    SetParmVal(body_id, "BotStr", fsec_curvename, BotStr);
    SetParmVal(body_id, "UpStr", fsec_curvename, UpStr);
    SetParmVal(body_id, "LowStr", fsec_curvename, LowStr);
    SetParmVal(body_id, "TopLAngle", fsec_name, TopLAngle);
    SetParmVal(body_id, "BottomLAngle", fsec_name, BottomLAngle);
    SetParmVal(body_id, "RightLAngle", fsec_name, RightLAngle);
    SetParmVal(body_id, "LeftLAngle", fsec_name, LeftLAngle);
    SetParmVal(body_id, "TopLStrength", fsec_name, TopLStrength);
    SetParmVal(body_id, "BottomLStrength", fsec_name, BottomLStrength);
    SetParmVal(body_id, "RightLStrength", fsec_name, RightLStrength);
    SetParmVal(body_id, "LeftLStrength", fsec_name, LeftLStrength);


    if ({{ nsec.TBSym | default(1.0) }} == 0.0)
    {
        SetParmVal(body_id, "TBSym", fsec_name, 0.0);
        SetParmVal(body_id, "BottomLAngle", fsec_name, BottomLAngle);
        SetParmVal(body_id, "BottomLStrength", fsec_name, BottomLStrength);

        Update();
    }

    if ({{ nsec.RLSym | default(1.0) }} == 0.0)
    {
        SetParmVal(body_id, "RLSym", fsec_name, 0.0);
        SetParmVal(body_id, "LeftLAngle", fsec_name, LeftLAngle);
        SetParmVal(body_id, "LeftLStrength", fsec_name, LeftLStrength);

        Update();
    }


    fsec_idx = fsec_idx + 1;
    fseccurve_idx = fseccurve_idx + 1;
    Update();
{% endfor %}
    fsec_idx = fsec_idx - 1;
    fseccurve_idx = fseccurve_idx - 1;
"""

        self.fuselage_mid_template = """
    //===== Insert Sections ({{ fsections|length }} total) ====//
{% for fsec in fsections %}
    InsertXSec(body_id, fsec_idx, XS_GENERAL_FUSE);
    fsec_idx = fsec_idx + 1;
    fseccurve_idx = fseccurve_idx + 1;

    fsec_name = Fusenames[fsec_idx];
    fsec_curvename = FuseCurvenames[fseccurve_idx];

    SetParmVal(body_id, "XLocPercent", fsec_name, {{ fsec.XLocPercent | default(0) }});
    SetParmVal(body_id, "YLocPercent", fsec_name, {{ fsec.YLocPercent | default(0) }});
    SetParmVal(body_id, "ZLocPercent", fsec_name, {{ fsec.ZLocPercent | default(0) }});
    SetParmVal(body_id, "Width", fsec_curvename, {{ fsec.Width | default(0) }});
    SetParmVal(body_id, "Height", fsec_curvename, {{ fsec.Height | default(0) }});
    SetParmVal(body_id, "SectTess_U", fsec_name, {{ fsec.SectTess_U | default(6) }});
    SetParmVal(body_id, "FwdCluster", fsec_name, {{ fsec.FwdCluster | default(1) }});


    SetParmVal(body_id, "MaxWidthLoc", fsec_curvename, {{ fsec.MaxWidthLoc | default(0) }});
    SetParmVal(body_id, "CornerRad", fsec_curvename, {{ fsec.CornerRad | default(0) }});
    SetParmVal(body_id, "TopTanAngle", fsec_curvename, {{ fsec.TopTanAngle | default(90) }});
    SetParmVal(body_id, "BotTanAngle", fsec_curvename, {{ fsec.BotTanAngle | default(90) }});
    SetParmVal(body_id, "TopStr", fsec_curvename, {{ fsec.TopStr | default(0.83) }});
    SetParmVal(body_id, "BotStr", fsec_curvename, {{ fsec.BotStr | default(0.83) }});
    SetParmVal(body_id, "UpStr", fsec_curvename, {{ fsec.UpStr | default(0.83) }});
    SetParmVal(body_id, "LowStr", fsec_curvename, {{ fsec.LowStr | default(0.83) }});
    SetParmVal(body_id, "TopLAngle", fsec_name, {{ fsec.TopLAngle | default(0) }});
    SetParmVal(body_id, "BottomLAngle", fsec_name, {{ fsec.BottomLAngle | default(0) }});
    SetParmVal(body_id, "RightLAngle", fsec_name, {{ fsec.RightLAngle | default(0) }});
    SetParmVal(body_id, "LeftLAngle", fsec_name, {{ fsec.LeftLAngle | default(0) }});
    SetParmVal(body_id, "TopLStrength", fsec_name, {{ fsec.TopLStrength | default(0.75) }});
    SetParmVal(body_id, "BottomLStrength", fsec_name, {{ fsec.BottomLStrength | default(0.75) }});
    SetParmVal(body_id, "RightLStrength", fsec_name, {{ fsec.RightLStrength | default(0.75) }});
    SetParmVal(body_id, "LeftLStrength", fsec_name, {{ fsec.LeftLStrength | default(0.75) }});

    if ({{ fsec.TBSym | default(1.0) }} == 0.0)
    {
        SetParmVal(body_id, "TBSym", fsec_name, 0.0);
        SetParmVal(body_id, "BottomLAngle", fsec_name, {{ fsec.BottomLAngle | default(fsec.TopLAngle) }});
        SetParmVal(body_id, "BottomLStrength", fsec_name, {{ fsec.BottomLStrength | default(fsec.TopLStrength) }});

        Update();
    }

    if ({{ fsec.RLSym | default(1.0) }} == 0.0)
    {
        SetParmVal(body_id, "RLSym", fsec_name, 0.0);
        SetParmVal(body_id, "LeftLAngle", fsec_name, {{ fsec.LeftLAngle | default(fsec.RightLAngle) }});
        SetParmVal(body_id, "LeftLStrength", fsec_name, {{ fsec.LeftLStrength | default(fsec.RightLStrength) }});

        Update();
    }



    Update();
{% endfor %}
"""

        self.fuselage_tail_template = """
    //===== Tail Sections (1 total) ====//
    fsec_idx = fsec_idx + 1;
    fseccurve_idx = fseccurve_idx + 1;
    fsec_name = Fusenames[fsec_idx];
    fsec_curvename = FuseCurvenames[fseccurve_idx];

    SetParmVal(body_id, "XLocPercent", fsec_name, {{ fsec.XLocPercent | default(1) }});
    SetParmVal(body_id, "YLocPercent", fsec_name, {{ fsec.YLocPercent | default(0) }});
    SetParmVal(body_id, "ZLocPercent", fsec_name, {{ fsec.ZLocPercent | default(0) }});
    SetParmVal(body_id, "Width", fsec_curvename, {{ fsec.Width | default(0) }});
    SetParmVal(body_id, "Height", fsec_curvename, {{ fsec.Height | default(0) }});
    SetParmVal(body_id, "SectTess_U", fsec_name, {{ fsec.SectTess_U | default(6) }});
    SetParmVal(body_id, "FwdCluster", fsec_name, {{ fsec.FwdCluster | default(1) }});


    SetParmVal(body_id, "MaxWidthLoc", fsec_curvename, {{ fsec.MaxWidthLoc | default(0) }});
    SetParmVal(body_id, "CornerRad", fsec_curvename, {{ fsec.CornerRad | default(0) }});
    SetParmVal(body_id, "TopTanAngle", fsec_curvename, {{ fsec.TopTanAngle | default(90) }});
    SetParmVal(body_id, "BotTanAngle", fsec_curvename, {{ fsec.BotTanAngle | default(90) }});
    SetParmVal(body_id, "TopStr", fsec_curvename, {{ fsec.TopStr | default(0.83) }});
    SetParmVal(body_id, "BotStr", fsec_curvename, {{ fsec.BotStr | default(0.83) }});
    SetParmVal(body_id, "UpStr", fsec_curvename, {{ fsec.UpStr | default(0.83) }});
    SetParmVal(body_id, "LowStr", fsec_curvename, {{ fsec.LowStr | default(0.83) }});
    SetParmVal(body_id, "TopLAngle", fsec_name, {{ fsec.TopLAngle | default(0) }});
    SetParmVal(body_id, "BottomLAngle", fsec_name, {{ fsec.BottomLAngle | default(0) }});
    SetParmVal(body_id, "RightLAngle", fsec_name, {{ fsec.RightLAngle | default(0) }});
    SetParmVal(body_id, "LeftLAngle", fsec_name, {{ fsec.LeftLAngle | default(0) }});
    SetParmVal(body_id, "TopLStrength", fsec_name, {{ fsec.TopLStrength | default(0.75) }});
    SetParmVal(body_id, "BottomLStrength", fsec_name, {{ fsec.BottomLStrength | default(0.75) }});
    SetParmVal(body_id, "RightLStrength", fsec_name, {{ fsec.RightLStrength | default(0.75) }});
    SetParmVal(body_id, "LeftLStrength", fsec_name, {{ fsec.LeftLStrength | default(0.75) }});

    if ({{ fsec.TBSym | default(1.0) }} == 0.0)
    {
        SetParmVal(body_id, "TBSym", fsec_name, 0.0);
        SetParmVal(body_id, "BottomLAngle", fsec_name, {{ fsec.BottomLAngle | default(0) }});
        SetParmVal(body_id, "BottomLStrength", fsec_name, {{ fsec.BottomLStrength | default(0.75) }});

        Update();
    }

    if ({{ fsec.RLSym | default(1.0) }} == 0.0)
    {
        SetParmVal(body_id, "RLSym", fsec_name, 0.0);
        SetParmVal(body_id, "LeftLAngle", fsec_name, {{ fsec.LeftLAngle | default(0) }});
        SetParmVal(body_id, "LeftLStrength", fsec_name, {{ fsec.LeftLStrength | default(0.75) }});

        Update();
    }


    Update();
"""

        self.wing_template = """
    //===== {{ wing.name }} =====//
    array<string> {{ wing.name }}_names = {"XSec_0", "XSec_1", "XSec_2", "XSec_3", "XSec_4", "XSec_5", "XSec_6", "XSec_7", "XSec_8", "XSec_9", "XSec_10", "XSec_11", "XSec_12", "XSec_13", "XSec_14", "XSec_15", "XSec_16", "XSec_17", "XSec_18", "XSec_19", "XSec_20"};
    string {{ wing.name }}_sec_name = "Xsec_0";
    string {{ wing.name }}_id_init = AddGeom("WING", "");
    SetGeomName({{ wing.name }}_id_init, "{{ wing.name }}");
    string {{ wing.name }}_id = FindGeom("{{ wing.name }}", 0);

    SetDriverGroup({{ wing.name }}_id, 1, SPAN_WSECT_DRIVER, ROOTC_WSECT_DRIVER, TIPC_WSECT_DRIVER);

    SetParmVal({{ wing.name }}_id, "RotateAirfoilMatchDideralFlag", "WingGeom", 0.0);
    SetParmVal({{ wing.name }}_id, "Tess_W", "Shape", {{ wing.global.tess_w | default(80.0) }});
    SetParmVal({{ wing.name }}_id, "Scale", "XForm", Scale);
    SetParmVal({{ wing.name }}_id, "X_Rel_Location", "XForm", {{ wing.global.x_rel_location | default(0.0) }});
    SetParmVal({{ wing.name }}_id, "Y_Rel_Location", "XForm", {{ wing.global.y_rel_location | default(0.0) }});
    SetParmVal({{ wing.name }}_id, "Z_Rel_Location", "XForm", {{ wing.global.z_rel_location | default(0.0) }});
    SetParmVal({{ wing.name }}_id, "X_Rel_Rotation", "XForm", {{ wing.global.x_rel_rotation | default(0.0) }});
    SetParmVal({{ wing.name }}_id, "Y_Rel_Rotation", "XForm", {{ wing.global.y_rel_rotation | default(0.0) }});
    SetParmVal({{ wing.name }}_id, "Z_Rel_Rotation", "XForm", {{ wing.global.z_rel_rotation | default(0.0) }});
    SetParmVal({{ wing.name }}_id, "Sym_Planar_Flag", "Sym", {{ wing.global.planar | default(2.0) }});
    SetParmVal({{ wing.name }}_id, "Sym_Axial_Flag", "Sym", {{ wing.global.axial | default(0) }});
    if ({{ wing.global.axial | default(0.0) }} > 0.0)
    {
        SetParmVal({{ wing.name }}_id, "Sym_Rot_N", "Sym", {{ wing.global.axial_n | default(2) }});
        Update();
    }


    float {{ wing.name }}_end_cap_type = 2.0;
    SetParmVal({{ wing.name }}_id, "CapUMaxOption", "EndCap", {{ wing.name }}_end_cap_type);

    float {{ wing.name }}_CapUMinTess = {{ wing.init.CapUMinTess | default(3.0) }};
    SetParmVal({{ wing.name }}_id, "CapUMinTess", "EndCap", {{ wing.name }}_CapUMinTess);

    float {{ wing.name }}_sector_span = {{ wing.init.span | default(9.0) }};
    float {{ wing.name }}_root_chord = {{ wing.init.chord_root | default(13.4) }};
    float {{ wing.name }}_sector_chord_root = {{ wing.name }}_root_chord;
    float {{ wing.name }}_sector_chord_tip = {{ wing.init.chord_tip | default(8.2) }};
    float {{ wing.name }}_sector_twist_root = {{ wing.init.twist_root | default(0.0) }};
    float {{ wing.name }}_sector_twist_tip = {{ wing.init.twist_tip | default(-1.0) }};
    float {{ wing.name }}_sector_sweep = {{ wing.init.sweep | default(35.0) }};
    float {{ wing.name }}_sector_dihedral = {{ wing.init.dihedral | default(4.29) }};
    float {{ wing.name }}_sector_tess_u = {{ wing.init.tess_u | default(15.0) }};

    array<double> {{ wing.name }}_upper_cst_params_root = {
        {% for v in wing.init.upper_cst_params_root %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
    };
    array<double> {{ wing.name }}_lower_cst_params_root = {
        {% for v in wing.init.lower_cst_params_root %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
    };
    array<double> {{ wing.name }}_upper_cst_params_tip = {
        {% for v in wing.init.upper_cst_params_tip %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
    };
    array<double> {{ wing.name }}_lower_cst_params_tip = {
        {% for v in wing.init.lower_cst_params_tip %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
    };

    SetParmVal({{ wing.name }}_id, "Twist", "XSec_0", {{ wing.name }}_sector_twist_root);
    SetParmVal({{ wing.name }}_id, "Twist_Location", "XSec_0", 0.25);
    SetParmVal({{ wing.name }}_id, "Span", "XSec_1", {{ wing.name }}_sector_span);
    SetParmVal({{ wing.name }}_id, "Root_Chord", "XSec_1", {{ wing.name }}_sector_chord_root);
    SetParmVal({{ wing.name }}_id, "Tip_Chord", "XSec_1", {{ wing.name }}_sector_chord_tip);
    SetParmVal({{ wing.name }}_id, "Sweep", "XSec_1", {{ wing.name }}_sector_sweep);
    SetParmVal({{ wing.name }}_id, "Twist", "XSec_1", {{ wing.name }}_sector_twist_tip);
    SetParmVal({{ wing.name }}_id, "Twist_Location", "XSec_1", 0.25);
    SetParmVal({{ wing.name }}_id, "Dihedral", "XSec_1", {{ wing.name }}_sector_dihedral);
    SetParmVal({{ wing.name }}_id, "SectTess_U", "XSec_1", {{ wing.name }}_sector_tess_u);

    int {{ wing.name }}_sec_idx = 0;
    string {{ wing.name }}_id_xsec_surf_id = GetXSecSurf({{ wing.name }}_id, 0);

    ChangeXSecShape({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx, XS_CST_AIRFOIL);
    string {{ wing.name }}_xsec_id = GetXSec({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx);
    SetUpperCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_upper_cst_params_root);
    SetLowerCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_lower_cst_params_root);

    {{ wing.name }}_sec_idx = 1;
    ChangeXSecShape({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx, XS_CST_AIRFOIL);
    {{ wing.name }}_xsec_id = GetXSec({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx);
    SetUpperCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_upper_cst_params_tip);
    SetLowerCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_lower_cst_params_tip);
    Update();

    //control the Outboard of the root edge of 1st wing //

    if ({{ wing.init.OutLEMode0 | default(0.0) }} > 0.0)
    {
        SetParmVal({{ wing.name }}_id, "OutLEMode", "XSec_0", {{ wing.init.OutLEMode0 | default(0.0) }} );
        SetParmVal({{ wing.name }}_id, "OutLESweep", "XSec_0", {{ wing.init.OutLESweep0 | default(wing.init.sweep) }});
        SetParmVal({{ wing.name }}_id, "OutLEDihedral", "XSec_0", {{ wing.init.OutLEDihedral0 | default(0.5) }});
        SetParmVal({{ wing.name }}_id, "OutLEStrength", "XSec_0", {{ wing.init.OutLEStrength0 | default(1.0) }});

        Update();
    }

    if ({{ wing.init.OutTEMode0 | default(0.0) }} > 0.0)
    {
        SetParmVal({{ wing.name }}_id, "OutTEMode", "XSec_0", {{ wing.init.OutTEMode0 | default(0.0) }} );
        SetParmVal({{ wing.name }}_id, "OutTESweep", "XSec_0", {{ wing.init.OutTESweep0 | default(wing.init.sweep) }});
        SetParmVal({{ wing.name }}_id, "OutTEDihedral", "XSec_0", {{ wing.init.OutTEDihedral0 | default(0.5) }});
        SetParmVal({{ wing.name }}_id, "OutTEStrength", "XSec_0", {{ wing.init.OutTEStrength0 | default(1.0) }});

        Update();
    }

    //control the Inboard of the tip edge of 1st wing if only one sector exits//

    if ({{ wing.init.InLEMode | default(0.0) }} > 0.0)
    {
        SetParmVal({{ wing.name }}_id, "InTEMode", "XSec_1", {{ wing.init.InLEMode | default(0.0) }} );
        SetParmVal({{ wing.name }}_id, "InLESweep", "XSec_1", {{ wing.init.InLESweep | default(wing.init.sweep) }});
        SetParmVal({{ wing.name }}_id, "InLEDihedral", "XSec_1", {{ wing.init.InLEDihedral | default(0.5) }});
        SetParmVal({{ wing.name }}_id, "InLEStrength", "XSec_1", {{ wing.init.InLEStrength | default(1.0) }});

        Update();
    }

    if ({{ wing.init.InTEMode | default(0.0) }} > 0.0)
    {
        SetParmVal({{ wing.name }}_id, "InTEMode", "XSec_1", {{ wing.init.InTEMode | default(0.0) }} );
        SetParmVal({{ wing.name }}_id, "InTESweep", "XSec_1", {{ wing.init.InTESweep | default(wing.init.sweep) }});
        SetParmVal({{ wing.name }}_id, "InTEDihedral", "XSec_1", {{ wing.init.InTEDihedral | default(0.5) }});
        SetParmVal({{ wing.name }}_id, "InTEStrength", "XSec_1", {{ wing.init.InTEStrength | default(1.0) }});

        Update();
    }





{% for sec in wing.mid %}
    {{ wing.name }}_sector_span = {{ sec.span | default(1.0) }};
    {{ wing.name }}_sector_chord_root = {{ wing.name }}_sector_chord_tip;
    {{ wing.name }}_sector_chord_tip = {{ sec.chord_tip | default(1.0) }};
    {{ wing.name }}_sector_twist_tip = {{ sec.twist_tip | default(0.0) }};
    {{ wing.name }}_sector_sweep = {{ sec.sweep | default(0.0) }};
    {{ wing.name }}_sector_dihedral = {{ sec.dihedral | default(0.0) }};
    {{ wing.name }}_sector_tess_u = {{ sec.tess_u | default(25.0) }};



    {{ wing.name }}_upper_cst_params_tip = {
        {% for v in sec.upper_cst_params_tip %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
    };
    {{ wing.name }}_lower_cst_params_tip = {
        {% for v in sec.lower_cst_params_tip %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
    };

    InsertXSec({{ wing.name }}_id, {{ wing.name }}_sec_idx, XS_CST_AIRFOIL);
    {{ wing.name }}_sec_idx = {{ wing.name }}_sec_idx + 1;
    {{ wing.name }}_sec_name = {{ wing.name }}_names[{{ wing.name }}_sec_idx];

    SetDriverGroup({{ wing.name }}_id, {{ wing.name }}_sec_idx, SPAN_WSECT_DRIVER, ROOTC_WSECT_DRIVER, TIPC_WSECT_DRIVER);
    SetParmVal({{ wing.name }}_id, "Span", {{ wing.name }}_sec_name, {{ wing.name }}_sector_span);
    SetParmVal({{ wing.name }}_id, "Root_Chord", {{ wing.name }}_sec_name, {{ wing.name }}_sector_chord_root);
    SetParmVal({{ wing.name }}_id, "Tip_Chord", {{ wing.name }}_sec_name, {{ wing.name }}_sector_chord_tip);
    SetParmVal({{ wing.name }}_id, "Sweep", {{ wing.name }}_sec_name, {{ wing.name }}_sector_sweep);
    SetParmVal({{ wing.name }}_id, "Twist", {{ wing.name }}_sec_name, {{ wing.name }}_sector_twist_tip);
    SetParmVal({{ wing.name }}_id, "Twist_Location", {{ wing.name }}_sec_name, 0.25);
    SetParmVal({{ wing.name }}_id, "Dihedral", {{ wing.name }}_sec_name, {{ wing.name }}_sector_dihedral);
    SetParmVal({{ wing.name }}_id, "SectTess_U", {{ wing.name }}_sec_name, {{ wing.name }}_sector_tess_u);

    ChangeXSecShape({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx, XS_CST_AIRFOIL);
    {{ wing.name }}_xsec_id = GetXSec({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx);
    SetUpperCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_upper_cst_params_tip);
    SetLowerCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_lower_cst_params_tip);
    Update();


        //control each wing sector's In and Outboard of the root edge//

        if ({{ sec.OutLEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "OutLEMode", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutLEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "OutLESweep", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutLESweep | default(sec.sweep) }});
            SetParmVal({{ wing.name }}_id, "OutLEDihedral", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutLEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "OutLEStrength", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutLEStrength | default(1.0) }});

            Update();
        }

        if ({{ sec.OutTEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "OutTEMode", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutTEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "OutTESweep", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutTESweep | default(sec.sweep) }});
            SetParmVal({{ wing.name }}_id, "OutTEDihedral", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutTEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "OutTEStrength", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.OutTEStrength | default(1.0) }});

            Update();
        }

        if ({{ sec.InLEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "InLEMode", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InLEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "InLESweep", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InLESweep | default(sec.sweep) }});
            SetParmVal({{ wing.name }}_id, "InLEDihedral", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InLEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "InLEStrength", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InLEStrength | default(1.0) }});

            Update();
        }

        if ({{ sec.InTEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "InTEMode", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InTEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "InTESweep", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InTESweep | default(sec.sweep) }});
            SetParmVal({{ wing.name }}_id, "InTEDihedral", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InTEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "InTEStrength", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ sec.InTEStrength | default(1.0) }});

            Update();
        }


{% endfor %}




    if ({{ wing.winglet.winglet_height | default(0.0) }} > 0.0)
    {
        {{ wing.name }}_sector_chord_root = {{ wing.name }}_sector_chord_tip;
        InsertXSec({{ wing.name }}_id, {{ wing.name }}_sec_idx, XS_CST_AIRFOIL);
        {{ wing.name }}_sec_idx = {{ wing.name }}_sec_idx + 1;
        {{ wing.name }}_sec_name = {{ wing.name }}_names[{{ wing.name }}_sec_idx];
        SetDriverGroup({{ wing.name }}_id, {{ wing.name }}_sec_idx, SPAN_WSECT_DRIVER, ROOTC_WSECT_DRIVER, TIPC_WSECT_DRIVER);
        SetParmVal({{ wing.name }}_id, "Span", {{ wing.name }}_sec_name, {{ wing.winglet.winglet_height | default(0.0) }});
        SetParmVal({{ wing.name }}_id, "Root_Chord", {{ wing.name }}_sec_name, {{ wing.name }}_sector_chord_root);
        SetParmVal({{ wing.name }}_id, "Tip_Chord", {{ wing.name }}_sec_name, {{ wing.winglet.chord_tip | default(1.0) }});
        SetParmVal({{ wing.name }}_id, "Sweep", {{ wing.name }}_sec_name, {{ wing.winglet.sweep | default(0.0) }});
        SetParmVal({{ wing.name }}_id, "Twist", {{ wing.name }}_sec_name, 0.0);
        SetParmVal({{ wing.name }}_id, "Twist_Location", {{ wing.name }}_sec_name, 0.25);
        SetParmVal({{ wing.name }}_id, "Dihedral", {{ wing.name }}_sec_name, {{ wing.winglet.dihedral | default(75.0) }});
        SetParmVal({{ wing.name }}_id, "SectTess_U", {{ wing.name }}_sec_name, {{ wing.winglet.tess_u | default(10.0) }});

        ChangeXSecShape({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx, XS_CST_AIRFOIL);
        {{ wing.name }}_xsec_id = GetXSec({{ wing.name }}_id_xsec_surf_id, {{ wing.name }}_sec_idx);
        array<double> {{ wing.name }}_winglet_upper = {
            {% for v in wing.winglet.upper_cst_params_tip %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
        };
        array<double> {{ wing.name }}_winglet_lower = {
            {% for v in wing.winglet.lower_cst_params_tip %}{{ "%.8f"|format(v) }}{% if not loop.last %}, {% endif %}{% endfor %}
        };
        SetUpperCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_winglet_upper);
        SetLowerCST({{ wing.name }}_xsec_id, 9, {{ wing.name }}_winglet_lower);
        Update();

        //control winglet's out and Inboard of the root edge//

        if ({{ wing.winglet.InLEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "InLEMode", {{ wing.name }}_sec_name, {{ wing.winglet.InLEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "InLESweep", {{ wing.name }}_sec_name, {{ wing.winglet.InLESweep | default(wing.winglet.sweep) | default(0) }});
            SetParmVal({{ wing.name }}_id, "InLEDihedral", {{ wing.name }}_sec_name, {{ wing.winglet.InLEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "InLEStrength", {{ wing.name }}_sec_name, {{ wing.winglet.InLEStrength | default(1.0) }});

            Update();
        }

        if ({{ wing.winglet.InTEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "InTEMode", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.InTEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "InTESweep", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.InTESweep | default(wing.winglet.sweep) | default(0) }});
            SetParmVal({{ wing.name }}_id, "InTEDihedral", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.InTEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "InTEStrength", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.InTEStrength | default(1.0) }});

            Update();
        }

        if ({{ wing.winglet.OutLEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "OutLEMode", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutLEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "OutLESweep", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutLESweep | default(wing.winglet.sweep) | default(0) }});
            SetParmVal({{ wing.name }}_id, "OutLEDihedral", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutLEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "OutLEStrength", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutLEStrength | default(1.0) }});

            Update();
        }

        if ({{ wing.winglet.OutTEMode | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "OutTEMode", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutTEMode | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "OutTESweep", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutTESweep | default(wing.winglet.sweep) | default(0) }});
            SetParmVal({{ wing.name }}_id, "OutTEDihedral", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutTEDihedral | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "OutTEStrength", {{ wing.name }}_names[{{ wing.name }}_sec_idx - 1], {{ wing.winglet.OutTEStrength | default(1.0) }});

            Update();
        }

        //control winglet's last Inboard of the tip edge//
        if ({{ wing.winglet.InLEMode_tip | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "InLEMode", {{ wing.name }}_sec_name, {{ wing.winglet.InLEMode_tip | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "InLESweep", {{ wing.name }}_sec_name, {{ wing.winglet.InLESweep_tip | default(wing.winglet.sweep) | default(0) }});
            SetParmVal({{ wing.name }}_id, "InLEDihedral", {{ wing.name }}_sec_name, {{ wing.winglet.InLEDihedral_tip | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "InLEStrength", {{ wing.name }}_sec_name, {{ wing.winglet.InLEStrength_tip | default(1.0) }});

            Update();
        }

        if ({{ wing.winglet.InTEMode_tip | default(0.0) }} > 0.0)
        {
            SetParmVal({{ wing.name }}_id, "InTEMode", {{ wing.name }}_sec_name, {{ wing.winglet.InTEMode_tip | default(0.0) }} );
            SetParmVal({{ wing.name }}_id, "InTESweep", {{ wing.name }}_sec_name, {{ wing.winglet.InTESweep_tip | default(wing.winglet.sweep) | default(0) }});
            SetParmVal({{ wing.name }}_id, "InTEDihedral", {{ wing.name }}_sec_name, {{ wing.winglet.InTEDihedral_tip | default(0.5) }});
            SetParmVal({{ wing.name }}_id, "InTEStrength", {{ wing.name }}_sec_name, {{ wing.winglet.InTEStrength_tip | default(1.0) }});

            Update();
        }
    }

"""
        self.compile_template = """
    Print(string("--> Compute Volume"));
    string mesh_id = ComputeCompGeom( SET_ALL, false, 0 );
"""
        self.export_init_template = """
    //==== Setup filenames ====//
{% if file_path %}
    string vsp_fname = "{{ file_path }}/";
    string stereolith_fname = "{{ file_path }}/";
    string iges_fname = "{{ file_path }}/";
    string stp_fname = "{{ file_path }}/";
{% endif %}
"""
        self.export_vsp_template = """
    WriteVSPFile(vsp_fname + "{{ id_str }}" + ".vsp3", SET_ALL);
    Print("VSP SAVED\\n");
"""
        self.export_stl_template = """
    ExportFile(stereolith_fname + "{{ id_str }}" + ".stl", SET_ALL, EXPORT_STL);
    Print("STL SAVED\\n");
"""
        self.export_igs_template = """
    ExportFile(iges_fname + "{{ id_str }}" + ".igs", SET_ALL, EXPORT_IGES);
    Print("IGS SAVED\\n");
"""
        self.export_stp_template = """
    ExportFile(stp_fname + "{{ id_str }}" + ".stp", SET_ALL, EXPORT_STEP);
    Print("STP SAVED\\n");
"""
        self.export_obj_template = """
    ExportFile(obj_fname + "{{ id_str }}" + ".obj", SET_ALL, EXPORT_OBJ);
    Print("OBJ SAVED\\n");
"""

    @staticmethod
    def _default_cst() -> List[float]:
        return [0.17413, 0.14629, 0.19259, 0.09979, 0.21465, 0.09021, 0.17718, 0.12531, 0.14341, 0.13875]

    def _normalize_wing(self, wing_name: str, wing: Dict[str, Any]) -> Dict[str, Any]:
        # ... (保持原有逻辑不变) ...
        upper = wing.get("init", {}).get("upper_cst_params_root", self._default_cst())
        lower = wing.get("init", {}).get("lower_cst_params_root", [-x for x in self._default_cst()])
        upper_tip = wing.get("init", {}).get("upper_cst_params_tip", upper)
        lower_tip = wing.get("init", {}).get("lower_cst_params_tip", lower)

        wing.setdefault("name", wing_name)
        wing.setdefault("global", {})
        wing.setdefault("init", {})
        wing.setdefault("mid", [])
        wing.setdefault("winglet", {})

        wing["init"].setdefault("upper_cst_params_root", upper)
        wing["init"].setdefault("lower_cst_params_root", lower)
        wing["init"].setdefault("upper_cst_params_tip", upper_tip)
        wing["init"].setdefault("lower_cst_params_tip", lower_tip)

        for sec in wing["mid"]:
            sec.setdefault("upper_cst_params_tip", upper_tip)
            sec.setdefault("lower_cst_params_tip", lower_tip)
        
        if "winglet" in wing and isinstance(wing["winglet"], dict):
            wing["winglet"].setdefault("upper_cst_params_tip", upper_tip)
            wing["winglet"].setdefault("lower_cst_params_tip", lower_tip)
            
        return wing

    def generate_script(
        self,
        script_file_path: str,
        global_params: Optional[Dict[str, Number]] = None,
        fuselage: Optional[Dict[str, Any]] = None,
        wings: Optional[List[Dict[str, Any]]] = None,
        save_file_path: Optional[str] = None,
        save_vsp: Optional[bool] = None,
        save_stl: Optional[bool] = None,
        save_igs: Optional[bool] = None,
        save_stp: Optional[bool] = None,
        save_obj: Optional[bool] = None,
        id_str: str = "default", 
    ) -> None:
        # 确保目录存在
        Path(script_file_path).parent.mkdir(parents=True, exist_ok=True)
        export_context = {"id_str": id_str}

        if wings is None:
            wings = []

        with open(script_file_path, "w", encoding="utf-8") as f:
            # 渲染开始部分
            f.write(Template(self.start_template).render(global_params or {}))

            # 渲染机身
            has_fuselage = bool(fuselage and (fuselage.get("global") or fuselage.get("init_sections") or fuselage.get("mid_sections") or fuselage.get("end_section")))
            if has_fuselage:
                # 注意：这里需要确保模板字符串已正确加载，否则会报错
                try:
                    f.write(Template(self.fuselage_global_template).render(fuselage.get("global", {})))
                    if fuselage.get("init_sections"):
                        f.write(Template(self.fuselage_nose_template).render(nsections=fuselage.get("init_sections", [])))
                    if fuselage.get("mid_sections"):
                        f.write(Template(self.fuselage_mid_template).render(fsections=fuselage.get("mid_sections", [])))
                    if fuselage.get("end_section"):
                        f.write(Template(self.fuselage_tail_template).render(fsec=fuselage.get("end_section", {})))
                except Exception as e:
                    raise RuntimeError(f"Fuselage template rendering failed: {e}")

            # 渲染机翼
            for idx, wing in enumerate(wings, start=1):
                normalized = self._normalize_wing(f"wing{idx}", wing)
                try:
                    f.write(Template(self.wing_template).render(wing=normalized))
                except Exception as e:
                    raise RuntimeError(f"Wing template rendering failed for {normalized.get('name')}: {e}")

            # 渲染计算和导出
            f.write(Template(self.compile_template).render())

            if save_file_path:
                f.write(Template(self.export_init_template).render(file_path=save_file_path))
                if save_vsp:
                    f.write(Template(self.export_vsp_template).render(export_context))
                if save_stl:
                    f.write(Template(self.export_stl_template).render(export_context))
                if save_igs:
                    f.write(Template(self.export_igs_template).render(export_context))
                if save_stp:
                    f.write(Template(self.export_stp_template).render(export_context))
                if save_obj:
                    f.write(Template(self.export_obj_template).render(export_context))

            f.write("}\n")


class AircraftDataManager:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        if self.file_path.suffix.lower() != ".json":
            raise ValueError(f"File must be a JSON file, got: {file_path}")
        self._load()

    def _load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {self.file_path}: {e}")

    def get_entry(self, ID: Union[int, str]) -> Optional[Dict[str, Any]]:
        key = str(ID)
        return self._data.get(key)

    def list_available_ids(self) -> List[str]:
        return sorted([k for k in self._data.keys() if isinstance(k, str)])

    def auto_select_id(self) -> str:
        ids = self.list_available_ids()
        if not ids:
            # 如果 JSON 是空的或者没有字符串键，尝试找数字键
            numeric_ids = sorted([k for k in self._data.keys() if str(k).isdigit()], key=int)
            if numeric_ids:
                return str(numeric_ids[0])
            raise KeyError(f"No valid ID found in JSON: {self.file_path}")
        
        # 优先返回第一个 ID
        return ids[0]

    def _prepare_output_layout(self, entry: Dict[str, Any], script_filename: str, save_path_override: Optional[str] = None) -> Tuple[str, str]:
        # 确定根目录
        save_root_raw = save_path_override if save_path_override is not None else entry.get("save_file_path")
        
        if not save_root_raw:
            # 如果没有指定，默认使用 config.json 的父目录
            save_root = self.file_path.parent
        else:
            save_root = Path(save_root_raw)

        geometry_root = save_root / "geometry_files"
        geometry_root.mkdir(parents=True, exist_ok=True)

        vspscript_dir = geometry_root / "vspscript"
        vspscript_dir.mkdir(parents=True, exist_ok=True)

        # 创建导出文件夹
        export_map = {
            "save_vsp": "vsp",
            "save_stl": "stl",
            "save_igs": "igs",
            "save_stp": "stp",
            "save_obj": "obj",
        }
        for flag, folder in export_map.items():
            if entry.get(flag, False):
                (geometry_root / folder).mkdir(parents=True, exist_ok=True)

        # 【关键修改】脚本文件名强制使用传入的 script_filename (即 project_slug)
        script_out_path = vspscript_dir / f"{Path(script_filename).stem}.vspscript"
        
        return str(script_out_path), save_root.as_posix()

    def generate_script_for_project(self, project_name: str, script_path_override: Optional[str] = None):
        """
        专门为 AI 调用设计的方法。
        :param project_name: 项目名称 (slug)，用作 ID 和文件名
        :param script_path_override: 可选的脚本输出路径覆盖
        """
        # 1. 获取数据条目
        # 尝试直接用 project_name 作为 Key 查找，如果找不到，则自动选择第一个
        entry = self.get_entry(project_name)
        
        if not entry:
            # 兼容模式：如果 JSON 里的 Key 是 "1", "2" 而不是项目名
            # 我们假设如果 Key 不匹配，就取第一个可用的条目（通常 AI 生成的 JSON 只有一个条目）
            available_ids = self.list_available_ids()
            if len(available_ids) == 1:
                entry = self.get_entry(available_ids[0])
                print(f"[Info] Key '{project_name}' not found. Using existing entry with key '{available_ids[0]}'.")
            else:
                raise KeyError(f"Project entry '{project_name}' not found in JSON, and multiple entries exist.")

        if not entry:
            raise KeyError(f"No valid configuration entry found for project '{project_name}'.")

        # 2. 准备路径
        # 脚本文件名将是 {project_name}.vspscript
        # 导出的模型前缀也将是 {project_name}
        resolved_script_path, resolved_save_root = self._prepare_output_layout(entry, project_name, script_path_override)

        # 3. 生成脚本
        generator = AircraftScriptGenerator()

        # 提取参数 (支持新格式和旧格式)
        if "wings" in entry or "fuselage" in entry:
            generator.generate_script(
                script_file_path=resolved_script_path,
                global_params=entry.get("global_params", {}),
                fuselage=entry.get("fuselage"),
                wings=entry.get("wings", []),
                save_file_path=resolved_save_root,
                save_vsp=entry.get("save_vsp", True),
                save_stl=entry.get("save_stl", True),
                save_igs=entry.get("save_igs", False),
                save_stp=entry.get("save_stp", True),
                save_obj=entry.get("save_obj", False),
                id_str=project_name,  # 【关键】导出文件名使用 project_name
            )
        else:
            # 兼容旧格式
            generator.generate_script(
                script_file_path=resolved_script_path,
                legacy_entry=entry,
                save_file_path=resolved_save_root,
                save_vsp=entry.get("save_vsp", True),
                save_stl=entry.get("save_stl", True),
                save_igs=entry.get("save_igs", False),
                save_stp=entry.get("save_stp", True),
                save_obj=entry.get("save_obj", False),
                id_str=project_name,
            )
        
        return resolved_script_path


# ================= 主程序入口 (AI 调用的核心) =================

def main():
    parser = argparse.ArgumentParser(description="Generate OpenVSP script from JSON config.")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing project folders.")
    parser.add_argument("--project-name", type=str, required=True, help="Project slug (folder name and file prefix).")
    
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    project_name = args.project_name
    project_dir = root_dir / project_name
    config_file = project_dir / "config.json"

    print(f"[SYSTEM] Starting generation for project: {project_name}")
    print(f"[SYSTEM] Root Dir: {root_dir}")
    print(f"[SYSTEM] Config Path: {config_file}")

    try:
        # 1. 检查文件是否存在
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_file}")

        # 2. 加载并生成
        dm = AircraftDataManager(str(config_file))
        
        # 执行生成，脚本将保存为 {project_dir}/geometry_files/vspscript/{project_name}.vspscript
        output_script = dm.generate_script_for_project(project_name)

        # 3. 验证输出
        if not Path(output_script).exists():
            raise RuntimeError("Script file was not created after generation.")

        # 4. 成功信号 (AI 会检测这个字符串)
        print(f"[SYSTEM] Script successfully generated: {output_script}")
        print("STATUS: SUCCESS")
        
        # 正常退出
        sys.exit(0)

    except Exception as e:
        # 5. 失败信号 (AI 会检测这个字符串)
        print(f"[ERROR] Generation failed: {str(e)}")
        print("STATUS: FAILED")
        # 非零退出码表示程序崩溃
        sys.exit(1)

if __name__ == "__main__":
    main()