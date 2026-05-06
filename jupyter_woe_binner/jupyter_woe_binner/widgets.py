#!/usr/bin/env python
# coding: utf-8

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Button, Output, HTML, Layout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ipyevents import Event
from .binning import calculate_woe_iv, calculate_spc_woe_iv

_C_CLR = {
    'good': '#5B8DB8',
    'bad': '#E07A5F',
    'rate': '#3D405B',
    'woe_pos': '#5B8DB8',
    'woe_neg': '#E07A5F',
    'spc': '#9B59B6',
    'bg': '#FAFAFA',
    'grid': '#E8E8E8',
    'text': '#2C3E50',
    'merge_dash': '#B0B0B0',
    'zero_line': '#999999',
}

_WOE_MERGE_THRESHOLD = 0.1

_CSS = """
<style>
.bw-btn {
    border-radius: 6px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif !important;
    transition: all 0.18s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    border: 1px solid #d0d5dd !important;
    padding: 3px 8px !important;
    min-width: 68px !important;
    font-size: 10.5px !important;
    line-height: 1.35 !important;
    position: relative !important;
    overflow: hidden !important;
}
.bw-btn:hover {
    transform: translateY(-0.5px);
    filter: brightness(0.97) !important;
    border-color: #b0b8c4 !important;
}
.bw-btn:active {
    transform: translateY(0.5px);
    filter: brightness(0.93) !important;
}

.bw-merge, .bw-split, .bw-confirm, .bw-reset, .bw-nav {
    background: #f0f2f5 !important;
    color: #2c3e6b !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.bw-merge:hover, .bw-split:hover, .bw-confirm:hover, .bw-reset:hover, .bw-nav:hover {
    background: #e8eaef !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06) !important;
}

.bw-boom {
    background: linear-gradient(135deg, #B84233, #D4543E) !important;
    color: rgba(255,255,255,0.95) !important;
    border: none !important;
    box-shadow: 0 1px 3px rgba(184,66,51,0.35), 0 1px 2px rgba(0,0,0,0.08) !important;
}
.bw-boom::after {
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important; bottom: 0 !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0) 100%) !important;
    pointer-events: none !important;
    border-radius: 6px !important;
}
.bw-boom:hover {
    box-shadow: 0 3px 10px rgba(184,66,51,0.4), 0 1px 3px rgba(0,0,0,0.1) !important;
    filter: brightness(1.08) saturate(1.1) !important;
}
.bw-boom:active {
    filter: brightness(0.95) saturate(0.95) !important;
}

.bw-divider {
    width: 1px;
    height: 24px;
    background: linear-gradient(180deg, transparent, #c8cdd5, transparent);
    margin: 0 3px;
    align-self: center;
}
</style>
"""


class BinningWidget:

    def __init__(self, df, var_name, target_name,
                 event_flag=1, non_event_flag=0,
                 initial_bins=None, max_bins=10, spc_values=None,
                 show_logo=True, min_split_pct=0.0):
        self.df = df
        self.var_name = var_name
        self.target_name = target_name
        self.event_flag = event_flag
        self.non_event_flag = non_event_flag
        self._max_bins = max_bins
        self._initial_bins_param = initial_bins
        self._show_logo = show_logo
        self.spc_values = spc_values if spc_values else []
        self.min_split_pct = min_split_pct

        self.series = df[var_name].copy()
        self.target = df[target_name].copy()

        self._normal_mask = pd.Series(True, index=self.series.index)
        for sv in self.spc_values:
            self._normal_mask &= (self.series != sv)

        self.normal_series = self.series[self._normal_mask]
        self.normal_target = self.target[self._normal_mask]

        if initial_bins is None:
            min_val = self.normal_series.min() if len(self.normal_series) > 0 else self.series.min()
            max_val = self.normal_series.max() if len(self.normal_series) > 0 else self.series.max()
            step = (max_val - min_val) / max_bins
            self.bins = [-np.inf] + [min_val + i * step for i in range(1, max_bins)] + [np.inf]
        else:
            self.bins = sorted(initial_bins)

        self._original_bins = self.bins[:]
        self.selected = []
        self._spc_count = len(self.spc_values)

        self._calc_stats()
        self._build_ui()

    def _calc_stats(self):
        normal_stats, normal_iv = calculate_woe_iv(
            self.normal_series, self.normal_target,
            self.bins, self.event_flag, self.non_event_flag)

        if self.spc_values:
            spc_df = calculate_spc_woe_iv(
                self.series, self.target, self.spc_values,
                self.event_flag, self.non_event_flag)
            n_spc = len(spc_df)
            n_norm = len(normal_stats)

            normal_stats['Index'] = range(n_spc, n_spc + n_norm)
            spc_df['Index'] = range(n_spc)

            self.stats_df = pd.concat([spc_df, normal_stats], ignore_index=True)
            self.total_iv = normal_iv + spc_df['IV'].sum()
        else:
            self.stats_df = normal_stats
            self.total_iv = normal_iv

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.fig = self._create_figure()
        self.table_output = Output()
        self.msg_output = Output()

        self.bin_selector = widgets.SelectMultiple(
            options=self._bin_options(),
            description='Select:',
            rows=min(8, len(self.stats_df)),
            layout=Layout(width='100%', height='130px'),
            style={'description_width': '56px'}
        )

        self.merge_btn = Button(description='⬌ Merge\n(Ctrl+⇧W)', layout=Layout(min_width='68px', height='36px'))
        self.merge_btn.add_class('bw-btn'); self.merge_btn.add_class('bw-merge')
        self.split_btn = Button(description='⬍ Split\n(Ctrl+⇧Q)', layout=Layout(min_width='68px', height='36px'))
        self.split_btn.add_class('bw-btn'); self.split_btn.add_class('bw-split')
        self.boom_btn = Button(description='💥 BOOM!\n(Ctrl+⇧B)', layout=Layout(min_width='68px', height='36px'))
        self.boom_btn.add_class('bw-btn'); self.boom_btn.add_class('bw-boom')
        self.confirm_btn = Button(description='✓ Confirm', layout=Layout(min_width='68px', height='36px'))
        self.confirm_btn.add_class('bw-btn'); self.confirm_btn.add_class('bw-confirm')
        self.reset_btn = Button(description='↺ Reset', layout=Layout(min_width='68px', height='36px'))
        self.reset_btn.add_class('bw-btn'); self.reset_btn.add_class('bw-reset')

        self.info_label = HTML(self._iv_html())
        self.selected_label = HTML(self._sel_html())

        self._setup_callbacks()
        self._setup_click_handler()

        divider1 = HTML('<div class="bw-divider"></div>')
        divider2 = HTML('<div class="bw-divider"></div>')
        btn_row = HBox(
            [self.merge_btn, self.split_btn, self.boom_btn,
             divider1, self.confirm_btn, divider2, self.reset_btn],
            layout=Layout(justify_content='center', align_items='center', gap='6px', padding='6px 0')
        )
        info_row = HBox([self.info_label, self.selected_label],
                        layout=Layout(justify_content='space-between', padding='2px 12px'))

        ui_children = [HTML(_CSS)]
        if self._show_logo:
            ui_children.append(HTML(
                '<div style="text-align:center;padding:8px 0 4px 0;">'
                '<span style="font-family:-apple-system,BlinkMacSystemFont,\'SF Pro Display\',sans-serif;'
                'font-size:22px;font-weight:300;color:#2C3E50;letter-spacing:3px;">'
                '— Binning it! —</span></div>'
            ))
            ui_children.append(HTML('<hr style="border:none;border-top:1px solid #e0e0e0;margin:0 16px;">'))
        ui_children.extend([
            self.bin_selector,
            btn_row,
            info_row,
            self.fig,
            self.msg_output,
            self.table_output,
        ])

        self.ui = VBox(ui_children, layout=Layout(padding='12px 16px', border='1px solid #e0e0e0', border_radius='14px',
                         background='#ffffff', max_width='1100px'))

        self.key_event = Event(source=self.ui, watched_events=['keydown'])
        self.key_event.on_dom_event(self._handle_key)
        self._update_table()

    def _bin_options(self):
        opts = []
        for i, row in self.stats_df.iterrows():
            prefix = '★' if i < self._spc_count else 'Bin'
            opts.append((f'{prefix} {i}: {row["Label"]}', i))
        return opts

    def _iv_html(self):
        return (f'<span style="font-family:-apple-system,sans-serif;font-size:13px;color:#2C3E50;">'
                f'<b>Total IV:</b> {self.total_iv:.4f}</span>')

    def _sel_html(self):
        t = ', '.join(str(s) for s in self.selected) if self.selected else 'none'
        return (f'<span style="font-family:-apple-system,sans-serif;font-size:13px;color:#86868b;">'
                f'Selected: {t}</span>')

    # -------------------------------------------------------------- Figure
    def _create_figure(self):
        stats = self.stats_df
        labels = stats['Label'].tolist()
        n = len(labels)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Distribution', 'Bad Rate', 'WOE'),
            column_widths=[0.35, 0.25, 0.40],
            horizontal_spacing=0.04,
        )

        good_pcts = (stats['High'] / (stats['High'] + stats['Low'])).fillna(0)
        bad_pcts = (stats['Low'] / (stats['High'] + stats['Low'])).fillna(0)

        good_colors = [_C_CLR['spc'] if i < self._spc_count else _C_CLR['good'] for i in range(n)]
        bad_colors = [_C_CLR['spc'] if i < self._spc_count else _C_CLR['bad'] for i in range(n)]

        fig.add_trace(go.Bar(
            y=labels, x=stats['High'], orientation='h', name='Good',
            marker_color=good_colors, marker_line_width=0.5, marker_line_color='white',
            text=[f'{p:.1%}' for p in good_pcts], textposition='inside',
            textfont=dict(color='white', size=10),
            hovertemplate='Good: %{x}<br>Pct: %{text}<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            y=labels, x=stats['Low'], orientation='h', name='Bad',
            marker_color=bad_colors, marker_line_width=0.5, marker_line_color='white',
            text=[f'{p:.1%}' for p in bad_pcts], textposition='inside',
            textfont=dict(color='white', size=10),
            hovertemplate='Bad: %{x}<br>Pct: %{text}<extra></extra>',
        ), row=1, col=1)

        rate_vals = stats['Low Rate'].tolist()
        rate_max = max(rate_vals) * 1.35 if rate_vals else 1.0
        fig.add_trace(go.Scatter(
            y=labels, x=rate_vals, mode='lines+markers+text',
            name='Bad Rate',
            line=dict(color=_C_CLR['rate'], width=2),
            marker=dict(size=7, color=_C_CLR['rate']),
            text=[f'{r:.1%}' for r in rate_vals], textposition='middle left',
            textfont=dict(color=_C_CLR['text'], size=10),
            hovertemplate='Bad Rate: %{x:.1%}<extra></extra>',
        ), row=1, col=2)

        woe = stats['WoE'].tolist()
        woe_colors = [_C_CLR['spc'] if i < self._spc_count else
                      (_C_CLR['woe_pos'] if w >= 0 else _C_CLR['woe_neg'])
                      for i, w in enumerate(woe)]
        woe_abs_max = max(abs(min(woe)), abs(max(woe)), 0.01) * 1.3
        fig.add_trace(go.Bar(
            y=labels, x=woe, orientation='h', name='WOE',
            marker_color=woe_colors, marker_line_width=0.5, marker_line_color='white',
            text=[f'{w:.3f}' for w in woe], textposition='outside',
            textfont=dict(color=_C_CLR['text'], size=10),
            hovertemplate='WOE: %{x:.3f}<extra></extra>',
        ), row=1, col=3)

        fig.add_shape(type='line', x0=0, x1=0, y0=-0.5, y1=n - 0.5,
                      line=dict(color=_C_CLR['zero_line'], width=1, dash='dot'),
                      row=1, col=3)

        for i in range(self._spc_count, n - 1):
            if abs(woe[i] - woe[i + 1]) < _WOE_MERGE_THRESHOLD:
                x_min = min(woe[i], woe[i + 1])
                x_max = max(woe[i], woe[i + 1])
                fig.add_shape(type='line',
                              x0=x_min, x1=x_max, y0=i + 0.5, y1=i + 0.5,
                              line=dict(color='#E74C3C', width=2, dash='dash'),
                              row=1, col=3)
                fig.add_annotation(
                    x=(x_min + x_max) / 2, y=i + 0.5,
                    text='⚡ merge', showarrow=True, arrowhead=2, arrowsize=0.8,
                    arrowcolor='#E74C3C', arrowwidth=1.5,
                    font=dict(size=11, color='#E74C3C', family='-apple-system,sans-serif'),
                    row=1, col=3)

        fig.update_layout(
            barmode='stack',
            showlegend=False,
            height=max(280, n * 38 + 60),
            hovermode='y unified',
            clickmode='event+select',
            plot_bgcolor=_C_CLR['bg'],
            paper_bgcolor='#ffffff',
            font=dict(family='-apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif',
                      size=11, color=_C_CLR['text']),
            margin=dict(l=100, r=30, t=50, b=40),
        )

        fig.update_yaxes(showgrid=False, row=1, col=1,
                         categoryorder='array', categoryarray=labels,
                         range=[n - 0.5, -0.5])
        fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=2,
                         categoryorder='array', categoryarray=labels,
                         range=[n - 0.5, -0.5])
        fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=3,
                         categoryorder='array', categoryarray=labels,
                         range=[n - 0.5, -0.5])
        fig.update_xaxes(showgrid=True, gridcolor=_C_CLR['grid'], row=1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor=_C_CLR['grid'], row=1, col=2)
        fig.update_xaxes(showgrid=True, gridcolor=_C_CLR['grid'], row=1, col=3)

        fig.update_xaxes(title_text='Count', row=1, col=1)
        fig.update_xaxes(title_text='Rate', row=1, col=2, tickformat='.0%',
                         range=[0, rate_max], constrain='range')
        fig.update_xaxes(title_text='WOE', row=1, col=3,
                         range=[-woe_abs_max, woe_abs_max], constrain='range')

        return go.FigureWidget(fig)

    # --------------------------------------------------------- Click handler
    def _setup_click_handler(self):
        for trace in self.fig.data:
            trace.on_click(lambda tr, pts, st: self._handle_bar_click(pts))

    def _handle_bar_click(self, points):
        if not points.point_inds:
            return
        idx = points.point_inds[0]
        if idx in self.selected:
            self.selected.remove(idx)
        else:
            self.selected.append(idx)
        self._update_selection()

    # --------------------------------------------------------- Merge (2+ consecutive)
    def _merge_selected(self):
        if len(self.selected) < 2:
            self._show_message('⚠️ 请选择两个或以上箱子进行合并')
            return

        sorted_sel = sorted(self.selected)

        if any(i < self._spc_count for i in sorted_sel):
            self._show_message('⚠️ 特殊值箱不可合并')
            return

        for i in range(len(sorted_sel) - 1):
            if sorted_sel[i + 1] - sorted_sel[i] != 1:
                self._show_message('⚠️ 选中的箱子必须连续，请重新选择')
                return

        norm_first = sorted_sel[0] - self._spc_count
        norm_last = sorted_sel[-1] - self._spc_count

        boundaries_to_remove = list(range(norm_first + 1, norm_last + 1))
        if not boundaries_to_remove:
            self._show_message('⚠️ 无需合并')
            return

        remaining = len(self.bins) - 1 - len(boundaries_to_remove)
        if remaining < 2:
            self._show_message('⚠️ 至少保留两个箱子')
            return

        for b in sorted(boundaries_to_remove, reverse=True):
            del self.bins[b]

        self.selected = []
        self._recalculate()
        self._show_message(f'✅ 合并成功，合并了 {len(boundaries_to_remove)} 个边界')

    # --------------------------------------------------------- Split (IV-optimal with min_split_pct)
    def _split_selected(self):
        if len(self.selected) != 1:
            self._show_message('⚠️ 请选择一个箱子进行拆分')
            return
        idx = self.selected[0]
        if idx < self._spc_count:
            self._show_message('⚠️ 特殊值箱不可拆分')
            return
        norm_idx = idx - self._spc_count
        lower, upper = self.bins[norm_idx], self.bins[norm_idx + 1]

        if np.isneginf(lower):
            mask = self.normal_series <= upper
        elif np.isposinf(upper):
            mask = self.normal_series > lower
        else:
            mask = (self.normal_series > lower) & (self.normal_series <= upper)

        data_in_bin = self.normal_series[mask]
        target_in_bin = self.normal_target[mask]
        if len(data_in_bin) == 0:
            self._show_message('⚠️ 该箱为空，无法拆分')
            return

        total_n = len(self.normal_series)
        total_event = (self.normal_target == self.event_flag).sum()
        total_nonevent = (self.normal_target == self.non_event_flag).sum()
        min_count = max(1, int(np.ceil(total_n * self.min_split_pct)))

        sorted_data = np.sort(data_in_bin.values)
        n_data = len(sorted_data)

        pcts = np.arange(1, 100) / 100.0
        candidate_splits = np.unique(np.percentile(sorted_data, pcts * 100))

        other_iv = self.total_iv
        if norm_idx < len(self.stats_df) - self._spc_count:
            bin_iv = self.stats_df.iloc[idx]['IV']
        else:
            bin_iv = 0
        other_iv -= bin_iv

        if self.spc_values:
            spc_df = calculate_spc_woe_iv(
                self.series, self.target, self.spc_values,
                self.event_flag, self.non_event_flag)
            other_iv -= spc_df['IV'].sum()
            other_iv += spc_df['IV'].sum()

        best_split, best_iv = None, self.total_iv
        min_val = 0.5

        for sv in candidate_splits:
            left_mask = data_in_bin <= sv
            right_mask = data_in_bin > sv
            left_count = left_mask.sum()
            right_count = right_mask.sum()

            if left_count < min_count or right_count < min_count:
                continue

            left_ev = (target_in_bin[left_mask] == self.event_flag).sum()
            left_nev = (target_in_bin[left_mask] == self.non_event_flag).sum()
            right_ev = (target_in_bin[right_mask] == self.event_flag).sum()
            right_nev = (target_in_bin[right_mask] == self.non_event_flag).sum()

            left_pct_ev = left_ev / total_event if total_event > 0 else 0
            left_pct_nev = left_nev / total_nonevent if total_nonevent > 0 else 0
            right_pct_ev = right_ev / total_event if total_event > 0 else 0
            right_pct_nev = right_nev / total_nonevent if total_nonevent > 0 else 0

            left_woe = np.log(
                (left_pct_nev + min_val / total_nonevent) /
                (left_pct_ev + min_val / total_event)
            ) if total_event > 0 and total_nonevent > 0 else 0
            right_woe = np.log(
                (right_pct_nev + min_val / total_nonevent) /
                (right_pct_ev + min_val / total_event)
            ) if total_event > 0 and total_nonevent > 0 else 0

            left_iv = (left_pct_nev - left_pct_ev) * left_woe
            right_iv = (right_pct_nev - right_pct_ev) * right_woe
            split_iv = other_iv - bin_iv + left_iv + right_iv

            if split_iv > best_iv:
                best_iv, best_split = split_iv, sv

        if best_split is None:
            if self.min_split_pct > 0:
                self._show_message(f'⚠️ 无法分裂（IV 未提升或分裂后存在箱占比 < {self.min_split_pct:.1%}）')
            else:
                self._show_message(f'⚠️ 无法分裂（IV {best_iv:.4f} ≤ 当前 {self.total_iv:.4f}）')
            return

        self.bins.insert(norm_idx + 1, best_split)
        self.selected = []
        self._recalculate()
        self._show_message(f'✅ 分裂成功，边界: {best_split:.2f}，IV: {best_iv:.4f}')

    # --------------------------------------------------------- BOOM! Split
    def _boom_split(self):
        if len(self.selected) != 1:
            self._show_message('💥 请选择一个箱子进行 BOOM! 分裂')
            return
        idx = self.selected[0]
        if idx < self._spc_count:
            self._show_message('⚠️ 特殊值箱不可拆分')
            return
        norm_idx = idx - self._spc_count
        lower, upper = self.bins[norm_idx], self.bins[norm_idx + 1]

        if np.isneginf(lower):
            mask = self.normal_series <= upper
        elif np.isposinf(upper):
            mask = self.normal_series > lower
        else:
            mask = (self.normal_series > lower) & (self.normal_series <= upper)

        data_in_bin = self.normal_series[mask]
        if len(data_in_bin) == 0:
            self._show_message('💥 该箱为空，无法分裂')
            return

        total_n = len(self.normal_series)
        min_count = max(1, int(np.ceil(total_n * 0.01)))
        unique_vals = np.sort(np.unique(data_in_bin))

        if len(unique_vals) < 2:
            self._show_message('💥 数据不足，无法分裂')
            return

        sorted_pcts = np.arange(1, 100) / 100.0
        percentiles = np.percentile(data_in_bin, sorted_pcts * 100)
        candidate_splits = sorted(set(np.round(percentiles, 8)))

        new_boundaries = []
        for sv in candidate_splits:
            if sv <= self.bins[norm_idx] or sv >= self.bins[norm_idx + 1]:
                continue
            if np.isneginf(lower) and sv <= lower:
                continue
            if np.isposinf(upper) and sv >= upper:
                continue
            new_boundaries.append(sv)

        if not new_boundaries:
            self._show_message('💥 无法找到有效分裂点')
            return

        test_bins = self.bins[:norm_idx + 1] + new_boundaries + self.bins[norm_idx + 1:]
        try:
            temp_stats, _ = calculate_woe_iv(
                self.normal_series, self.normal_target, test_bins,
                self.event_flag, self.non_event_flag)
        except Exception:
            self._show_message('💥 分裂计算失败')
            return

        valid_boundaries = []
        for i, sv in enumerate(new_boundaries):
            row_idx = i + 1
            if row_idx < len(temp_stats):
                bin_count = temp_stats.iloc[row_idx]['Total']
                if bin_count >= min_count:
                    valid_boundaries.append(sv)

        first_bin_count = temp_stats.iloc[0]['Total'] if len(temp_stats) > 0 else 0
        if first_bin_count < min_count:
            if valid_boundaries:
                valid_boundaries = valid_boundaries[1:]

        if not valid_boundaries:
            self._show_message('💥 分裂后所有箱占比均 < 1%，无法分裂')
            return

        final_bins = self.bins[:norm_idx + 1] + valid_boundaries + self.bins[norm_idx + 1:]
        self.bins = final_bins
        self.selected = []
        self._recalculate()
        self._show_message(f'💥 BOOM! 分裂成功，新增 {len(valid_boundaries)} 个边界，共 {len(self.bins) - 1} 个箱')

    # -------------------------------------------------------- Recalculate
    def _recalculate(self):
        try:
            self._calc_stats()
        except Exception as e:
            self._show_message(f'❌ 计算失败: {e}')
            return
        self._rebuild_figure()

    def _rebuild_figure(self):
        fig_idx = None
        for i, child in enumerate(self.ui.children):
            if child is self.fig:
                fig_idx = i
                break

        self.fig = self._create_figure()
        self._setup_click_handler()

        if fig_idx is not None:
            children = list(self.ui.children)
            children[fig_idx] = self.fig
            self.ui.children = tuple(children)

        self.bin_selector.options = self._bin_options()
        self.bin_selector.value = ()
        self.info_label.value = self._iv_html()
        self.selected_label.value = self._sel_html()
        self._update_table()

    # -------------------------------------------------------- Selection
    def _update_selection(self):
        sel = self.selected if self.selected else None
        for trace in self.fig.data:
            trace.selectedpoints = sel
        self.bin_selector.value = tuple(self.selected)
        self.selected_label.value = self._sel_html()

    # -------------------------------------------------------- Table
    def _update_table(self):
        with self.table_output:
            self.table_output.clear_output(wait=True)
            styled = self.stats_df.style.format({
                '%Total': '{:.1%}', '% High': '{:.1%}', '% Low': '{:.1%}',
                'Low Rate': '{:.1%}', 'WoE': '{:.3f}', 'IV': '{:.4f}', 'Odds': '{:.2f}'
            })
            styled = styled.set_properties(**{
                'font-size': '11px',
                'padding': '2px 6px',
                'white-space': 'nowrap',
                'font-family': '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif'
            })
            styled = styled.set_table_styles([
                {'selector': 'th', 'props': [
                    ('font-size', '12px'),
                    ('padding', '3px 6px'),
                    ('white-space', 'nowrap'),
                    ('text-align', 'center'),
                    ('font-family', '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif'),
                ]},
                {'selector': 'td', 'props': [
                    ('text-align', 'right'),
                ]},
                {'selector': 'th.col_heading', 'props': [
                    ('text-align', 'center'),
                ]},
            ], overwrite=False)
            styled = styled.bar(subset=['%Total'], color=_C_CLR['good'], vmin=0, vmax=1)
            try:
                woe_max = max(abs(self.stats_df['WoE'].min()), abs(self.stats_df['WoE'].max()), 0.01)
                styled = styled.bar(subset=['WoE'], align='mid',
                                    color=[_C_CLR['woe_neg'], _C_CLR['woe_pos']],
                                    vmin=-woe_max, vmax=woe_max)
            except Exception:
                pass
            if self._spc_count > 0:
                styled = styled.set_properties(
                    subset=pd.IndexSlice[:self._spc_count - 1, :],
                    **{'background-color': '#F0E6F6'})
            try:
                display(styled.hide(axis='index'))
            except AttributeError:
                display(styled)

    # -------------------------------------------------------- Callbacks
    def _setup_callbacks(self):
        self.merge_btn.on_click(lambda b: self._merge_selected())
        self.split_btn.on_click(lambda b: self._split_selected())
        self.boom_btn.on_click(lambda b: self._boom_split())
        self.confirm_btn.on_click(lambda b: self._confirm_binning())
        self.reset_btn.on_click(lambda b: self._reset())
        self.bin_selector.observe(self._on_bin_select, names='value')

    def _on_bin_select(self, change):
        self.selected = list(self.bin_selector.value)
        self._update_selection()

    def _handle_key(self, event):
        if event.get('ctrlKey') and event.get('shiftKey'):
            key = event.get('key', '').lower()
            if key == 'w':
                self._merge_selected()
            elif key == 'q':
                self._split_selected()
            elif key == 'b':
                self._boom_split()

    # -------------------------------------------------------- Actions
    def _confirm_binning(self):
        bin_dict = {self.var_name: self.bins}
        if self.spc_values:
            bin_dict[self.var_name + '_spc'] = self.spc_values
        print('✅ 分箱结果已确认：')
        print(bin_dict)
        self.output_bins = bin_dict
        return bin_dict

    def _reset(self):
        self.selected = []
        self.bins = self._original_bins[:]
        self._recalculate()
        self._show_message('↺ 已重置为初始状态')

    def _show_message(self, msg):
        with self.msg_output:
            self.msg_output.clear_output(wait=True)
            display(HTML(
                f'<div style="font-family:-apple-system,sans-serif;font-size:12px;'
                f'padding:5px 12px;border-radius:8px;background:#f5f5f7;color:#2C3E50;'
                f'margin:2px 0;">{msg}</div>'))

    def display(self):
        return self.ui


class BinningWidgetList:

    def __init__(self, df, var_name, target_name,
                 event_flag=1, non_event_flag=0, max_bins=10,
                 spc_values=None, min_split_pct=0.0, initial_bin_dir=None):
        if isinstance(var_name, str):
            var_name = [var_name]
        self.df = df
        self.var_names = var_name
        self.target_name = target_name
        self.event_flag = event_flag
        self.non_event_flag = non_event_flag
        self.max_bins = max_bins
        self.spc_values = spc_values if spc_values else []
        self.min_split_pct = min_split_pct
        self.initial_bin_dir = initial_bin_dir if initial_bin_dir else {}
        self._current_idx = 0
        self._confirmed_bins = {}

        self._widgets = {}
        for vn in self.var_names:
            ib = self.initial_bin_dir.get(vn, None)
            self._widgets[vn] = BinningWidget(
                df, var_name=vn, target_name=target_name,
                event_flag=event_flag, non_event_flag=non_event_flag,
                initial_bins=ib, max_bins=max_bins, spc_values=self.spc_values,
                show_logo=False, min_split_pct=min_split_pct
            )

        self._build_ui()

    def _build_ui(self):
        self.nav_label = HTML(self._nav_html())

        self.last_btn = Button(description='◀ Last', layout=Layout(min_width='60px', height='28px'))
        self.last_btn.add_class('bw-btn'); self.last_btn.add_class('bw-nav')
        self.next_btn = Button(description='Next ▶', layout=Layout(min_width='60px', height='28px'))
        self.next_btn.add_class('bw-btn'); self.next_btn.add_class('bw-nav')
        self.confirm_list_btn = Button(description='✓ Confirm', layout=Layout(min_width='68px', height='28px'))
        self.confirm_list_btn.add_class('bw-btn'); self.confirm_list_btn.add_class('bw-confirm')

        self.last_btn.on_click(lambda b: self._go_last())
        self.next_btn.on_click(lambda b: self._go_next())
        self.confirm_list_btn.on_click(lambda b: self._confirm_current())

        nav_row = HBox(
            [self.last_btn, self.nav_label, self.next_btn, self.confirm_list_btn],
            layout=Layout(justify_content='center', gap='12px', padding='6px 0')
        )

        self._content = VBox([self._current_widget().ui])

        self.ui = VBox([
            HTML(_CSS),
            HTML('<div style="text-align:center;padding:8px 0 2px 0;">'
                 '<span style="font-family:-apple-system,sans-serif;font-size:20px;font-weight:300;'
                 'color:#2C3E50;letter-spacing:2px;">— Binning it! —</span></div>'),
            HTML('<hr style="border:none;border-top:1px solid #e0e0e0;margin:0 16px;">'),
            nav_row,
            self._content,
        ], layout=Layout(padding='12px 16px', border='1px solid #e0e0e0', border_radius='14px',
                         background='#ffffff', max_width='1100px'))

    def _nav_html(self):
        vn = self.var_names[self._current_idx]
        confirmed_mark = ' ✓' if vn in self._confirmed_bins else ''
        return (f'<span style="font-family:-apple-system,sans-serif;font-size:13px;color:#2C3E50;'
                f'min-width:260px;display:inline-block;text-align:center;">'
                f'[{self._current_idx + 1}/{len(self.var_names)}] '
                f'<b>{vn}</b>{confirmed_mark}</span>')

    def _current_widget(self):
        return self._widgets[self.var_names[self._current_idx]]

    def _switch_to(self, idx):
        self._current_idx = idx
        self._content.children = (self._current_widget().ui,)
        self.nav_label.value = self._nav_html()

    def _go_next(self):
        if self._current_idx < len(self.var_names) - 1:
            self._switch_to(self._current_idx + 1)

    def _go_last(self):
        if self._current_idx > 0:
            self._switch_to(self._current_idx - 1)

    def _confirm_current(self):
        vn = self.var_names[self._current_idx]
        w = self._current_widget()
        self._confirmed_bins[vn] = w.bins[:]
        self.nav_label.value = self._nav_html()
        print(f'✅ {vn} 分箱已确认: {w.bins}')

    @property
    def bins(self):
        result = {}
        for vn in self.var_names:
            if vn in self._confirmed_bins:
                result[vn] = self._confirmed_bins[vn]
            else:
                result[vn] = self._widgets[vn].bins[:]
        if self.spc_values:
            result['_spc_values'] = self.spc_values
        return result

    def display(self):
        return self.ui
