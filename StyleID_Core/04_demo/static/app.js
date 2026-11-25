/* Giao diện demo: React + Ant Design, viết bằng htm nên không cần bước build. */
(function () {
    'use strict';
    const { useState, useEffect, useRef, useMemo } = React;
    const { ConfigProvider, theme, Card, Button, Segmented, Select, Slider, InputNumber,
        Tag, Table, Alert, Spin, Upload } = antd;
    const html = htm.bind(React.createElement);

    const PRESETS = [
        { id: 'lfw-1342', title: 'Cùng người', note: 'Một ảnh đeo kính râm' },
        { id: 'lfw-0557', title: 'Khác người', note: 'Cặp khó' },
        { id: 'lfw-0000', title: 'Cùng người', note: 'Cặp đối chứng' },
    ];
    const MODES = {
        dual_hp: 'Dual-Branch + Hard-Positive',
        dual: 'Dual-Branch',
        raw: 'Disentangle ID512',
    };
    const DEFAULT_THRESHOLD = 0.3;
    const THEME_KEY = 'styleid-theme';

    const ICONS = {
        sun: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
        moon: 'M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z',
        system: 'M3 5h18v11H3zM8 20h8M12 16v4',
        image: 'M4 4h16v16H4zM4 16l4.5-4.5 4 4L15 13l5 5M15.5 8.5h.01',
        faces: 'M9 3H5a2 2 0 0 0-2 2v4M15 3h4a2 2 0 0 1 2 2v4M9 21H5a2 2 0 0 1-2-2v-4M15 21h4a2 2 0 0 0 2-2v-4M9 10h.01M15 10h.01M9 15c1.8 1.4 4.2 1.4 6 0',
    };
    const icon = (name, size = 16) => html`
        <svg width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d=${ICONS[name]} />
        </svg>`;

    const signed = (value, digits) => `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;

    function useThemeMode() {
        const query = useMemo(() => matchMedia('(prefers-color-scheme: dark)'), []);
        const [mode, setMode] = useState(() => {
            try {
                const saved = localStorage.getItem(THEME_KEY);
                return ['light', 'dark'].includes(saved) ? saved : 'system';
            } catch (_) { return 'system'; }
        });
        const [systemDark, setSystemDark] = useState(query.matches);
        useEffect(() => {
            const onChange = event => setSystemDark(event.matches);
            query.addEventListener('change', onChange);
            return () => query.removeEventListener('change', onChange);
        }, [query]);
        const dark = mode === 'dark' || (mode === 'system' && systemDark);
        useEffect(() => {
            document.documentElement.dataset.theme = dark ? 'dark' : 'light';
            try { localStorage.setItem(THEME_KEY, mode); } catch (_) {}
        }, [dark, mode]);
        return { mode, setMode, dark };
    }

    function ImageCard({ title, src, disabled, onPick, onClear }) {
        const beforeUpload = file => {
            onPick(file);
            return Upload.LIST_IGNORE;
        };
        return html`
            <${Card} title=${title} extra=${src && html`
                <${Button} size="small" type="text" disabled=${disabled} onClick=${onClear}>Xóa<//>`}>
                <${Upload.Dragger} accept="image/*" showUploadList=${false} disabled=${disabled}
                    beforeUpload=${beforeUpload}>
                    <div className="drop-body">
                        ${src ? html`<img src=${src} alt=${title} />` : html`
                            ${icon('image', 36)}
                            <div className="drop-title">Chọn hoặc kéo thả ảnh vào đây</div>
                            <div className="muted">JPG, PNG, WebP</div>`}
                    </div>
                <//>
            <//>`;
    }

    function ScoreCard({ role, title, dim, score, threshold, parts, truth, ours }) {
        const same = score >= threshold;
        const percent = value => Math.max(0, Math.min(100, (value + 1) * 50));
        return html`
            <${Card}>
                <div className="card-head">
                    <div>
                        <div className="role">${role}</div>
                        <h2>${title}</h2>
                    </div>
                    <${Tag} bordered=${false}>${dim}<//>
                </div>
                <div className=${`score num${ours ? ' ours' : ''}`}>${score.toFixed(4)}</div>
                <div className="tags">
                    <${Tag} color=${same ? 'success' : 'warning'}>
                        ${same ? 'Cùng danh tính (s ≥ ngưỡng)' : 'Khác danh tính (s < ngưỡng)'}
                    <//>
                    ${truth !== null && html`
                        <${Tag} color=${same === truth ? 'processing' : 'error'}>
                            ${same === truth ? 'Đúng nhãn' : 'Sai nhãn'}
                        <//>`}
                </div>
                <div className="parts num">${parts}</div>
                <div className="scale">
                    <div className=${`scale-fill${ours ? ' ours' : ''}`} style=${{ width: `${percent(score)}%` }}></div>
                    <div className="scale-mark" style=${{ left: `${percent(threshold)}%` }}>
                        <span className="num">Ngưỡng ${threshold.toFixed(2)}</span>
                    </div>
                </div>
                <div className="scale-ends num"><span>-1</span><span>Cosine similarity</span><span>+1</span></div>
            <//>`;
    }

    function Results({ result, mode, threshold, truth, boxRef }) {
        const selected = result.modes[mode];
        const base = result.styleid.similarity;
        const score = selected.similarity;
        const label = value => value >= threshold ? 'Cùng danh tính' : 'Khác danh tính';
        const dual = selected.dimension === 1280;
        const component = mode === 'dual_hp' ? result.id_hard_positive : result.soup_raw;
        const parts = dual
            ? `StyleID: ${base.toFixed(4)} · ID${mode === 'dual_hp' ? ' + HP' : ''}: ${component.similarity.toFixed(4)} · 50/50`
            : `ID512: ${result.soup_raw.similarity.toFixed(4)}`;
        const gap = selected.dimension - 768;
        const rows = [
            ['Cosine similarity', base.toFixed(4), score.toFixed(4), signed(score - base, 4)],
            [`Phán đoán tại ngưỡng ${threshold.toFixed(2)}`, label(base), label(score),
                label(base) === label(score) ? 'Hai mô hình đồng thuận' : 'Hai mô hình phân kỳ'],
            ['Kích thước vector so sánh', '768 chiều (StyleID)', selected.dim, `${signed(gap, 0)} chiều`],
            ['Cách tính biểu diễn', 'Không phân rã (học gộp)', selected.mechanism,
                dual ? 'Kết hợp hai nhánh' : 'Nhánh danh tính'],
        ].map(([metric, a, b, delta]) => ({ key: metric, metric, a, b, delta }));
        const columns = [
            { title: 'Chỉ số', dataIndex: 'metric' },
            { title: 'StyleID (đối chứng)', dataIndex: 'a', className: 'num' },
            { title: selected.name, dataIndex: 'b', className: 'num' },
            { title: 'Chênh lệch (Δ)', dataIndex: 'delta', align: 'right', className: 'num' },
        ];
        return html`
            <div className="pair" ref=${boxRef}>
                <${ScoreCard} role="Mô hình đối chứng" title="StyleID (kwanY et al., SIGGRAPH 2026)"
                    dim="768 chiều" score=${base} threshold=${threshold} truth=${truth}
                    parts="Nhánh StyleID gốc" />
                <${ScoreCard} ours role="Mô hình đề tài" title=${selected.name} dim=${selected.dim}
                    score=${score} threshold=${threshold} truth=${truth} parts=${parts} />
            </div>
            <${Card} title="Bảng thông số đối chiếu thực nghiệm">
                <${Table} columns=${columns} dataSource=${rows} pagination=${false}
                    size="middle" scroll=${{ x: 720 }} />
            <//>`;
    }

    function Demo({ themeMode, setThemeMode }) {
        const [images, setImages] = useState([null, null]);
        const [preset, setPreset] = useState(null);
        const [mode, setMode] = useState('dual');
        const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
        const [result, setResult] = useState(null);
        const [device, setDevice] = useState(null);
        const [busy, setBusy] = useState(false);
        const [error, setError] = useState(null);
        const version = useRef(0);
        const resultBox = useRef(null);

        useEffect(() => {
            if (!result || !resultBox.current) return;
            const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
            resultBox.current.scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'nearest' });
        }, [result]);

        function setCustomImage(index, data) {
            version.current += 1;
            setImages(current => current.map((item, i) => (i === index ? data : item)));
            setPreset(null);
            setResult(null);
            setError(null);
            setMode(current => (current === 'dual_hp' ? 'dual' : current));
        }

        function pickFile(index, file) {
            if (!file.type.startsWith('image/')) {
                setError('Tệp đã chọn không phải là ảnh.');
                return;
            }
            const reader = new FileReader();
            reader.onload = () => setCustomImage(index, reader.result);
            reader.onerror = () => setError('Không đọc được tệp ảnh.');
            reader.readAsDataURL(file);
        }

        async function readJson(response) {
            let data;
            try { data = await response.json(); } catch (_) { throw new Error(`Mã lỗi ${response.status}`); }
            if (!response.ok || data.error) {
                const detail = data.detail || data.error;
                throw new Error(typeof detail === 'string' ? detail : `Mã lỗi ${response.status}`);
            }
            return data;
        }

        async function compare(nextMode, presetId) {
            const started = version.current;
            const payload = presetId ? { preset_id: presetId }
                : { img1_base64: images[0], img2_base64: images[1] };
            setBusy(true);
            setError(null);
            setResult(null);
            try {
                const data = await readJson(await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...payload, threshold, mode: nextMode }),
                }));
                if (started !== version.current) return;
                setDevice(data.device);
                setResult(data);
            } catch (reason) {
                if (started === version.current) setError(`Lỗi khi xử lý: ${reason.message}`);
            } finally {
                setBusy(false);
            }
        }

        async function choosePreset(id) {
            const started = ++version.current;
            setBusy(true);
            setError(null);
            try {
                const data = await readJson(await fetch(`/api/preset/${id}`));
                if (started !== version.current) { setBusy(false); return; }
                setImages([data.img1, data.img2]);
                setPreset(data);
                setMode('dual_hp');
            } catch (reason) {
                if (started === version.current) setError(`Không nạp được cặp LFW: ${reason.message}`);
                setBusy(false);
                return;
            }
            await compare('dual_hp', id);
        }

        function changeMode(next) {
            setMode(next);
            // Dùng lại điểm đã tính nếu kết quả hiện tại có chế độ này.
            if (result && !result.modes[next]) compare(next, preset && preset.id);
        }

        const ready = images[0] && images[1];
        const truth = preset ? preset.ground_truth === 'Cùng danh tính' : null;
        const themeOption = (value, iconName, label) => ({
            value, label: html`<span className="theme-option">${icon(iconName)}${label}</span>`,
        });
        const updateThreshold = value => { if (typeof value === 'number') setThreshold(value); };

        return html`
            <header className="top"><div className="wrap">
                <div className="brand">
                    ${icon('faces', 34)}
                    <div>
                        <h1>StyleID và Dual-Branch + Hard-Positive</h1>
                        <p>Demo cấu hình cuối LFW từ notebook kiểm chứng Colab</p>
                    </div>
                </div>
                <div className="top-tools">
                    ${device && html`<${Tag} bordered=${false}>${device}<//>`}
                    <${Segmented} aria-label="Giao diện" value=${themeMode} onChange=${setThemeMode} options=${[
                        themeOption('light', 'sun', 'Sáng'),
                        themeOption('dark', 'moon', 'Tối'),
                        themeOption('system', 'system', 'Hệ thống'),
                    ]} />
                </div>
            </div></header>

            <main className="wrap">
                <${Card} title="Cặp ảnh mẫu LFW View 2" extra=${preset && html`
                    <${Tag} color=${truth ? 'success' : 'warning'}>Nhãn: ${preset.ground_truth}<//>`}>
                    <div className="presets">
                        ${PRESETS.map(item => html`
                            <button type="button" key=${item.id} className="preset" disabled=${busy}
                                aria-pressed=${Boolean(preset && preset.id === item.id)}
                                onClick=${() => choosePreset(item.id)}>
                                <strong>${item.title}</strong><span>${item.note} · ${item.id}</span>
                            </button>`)}
                    </div>
                <//>

                <div className="pair">
                    ${['Ảnh 1: Khuôn mặt tham chiếu', 'Ảnh 2: Khuôn mặt đối chiếu'].map((title, index) => html`
                        <${ImageCard} key=${index} title=${title} src=${images[index]} disabled=${busy}
                            onPick=${file => pickFile(index, file)}
                            onClear=${() => setCustomImage(index, null)} />`)}
                </div>

                <${Card}>
                    <div className="controls">
                        <div>
                            <div className="field-head">
                                <span>Ngưỡng phân loại</span>
                                <${InputNumber} aria-label="Ngưỡng phân loại" size="small" min=${0.1} max=${0.9}
                                    step=${0.01} precision=${2} value=${threshold} onChange=${updateThreshold} />
                            </div>
                            <${Slider} min=${0.1} max=${0.9} step=${0.01} value=${threshold}
                                onChange=${updateThreshold} tooltip=${{ open: false }}
                                marks=${{ 0.1: '0.10', 0.3: '0.30', 0.9: '0.90' }} />
                        </div>
                        <div>
                            <div className="field-head"><span>Cấu hình so sánh</span></div>
                            <${Select} aria-label="Cấu hình so sánh" style=${{ width: '100%' }} size="large"
                                value=${mode} disabled=${busy} onChange=${changeMode}
                                options=${Object.entries(MODES).map(([value, label]) => ({
                                    value, label, disabled: value === 'dual_hp' && !preset,
                                }))} />
                        </div>
                        <${Button} type="primary" size="large" loading=${busy} disabled=${!ready}
                            onClick=${() => compare(mode, preset && preset.id)}>So sánh<//>
                    </div>
                    <p className="hint">Hard-Positive chỉ dùng cho các cặp mẫu LFW. Ngưỡng mặc định 0,30 tương ứng cột Acc@0.3 trong bảng kiểm chứng.</p>
                <//>

                ${error && html`<${Alert} type="error" showIcon closable message=${error}
                    onClose=${() => setError(null)} />`}
                ${busy && html`<div className="loading" role="status">
                    <${Spin} size="large" />
                    <p className="muted">Đang nạp dữ liệu hoặc tính điểm. Lần đầu cần tải khoảng 2 GB, xem tiến độ trong terminal.</p>
                </div>`}
                ${result && result.modes[mode] && html`<${Results} result=${result} mode=${mode}
                    threshold=${threshold} truth=${truth} boxRef=${resultBox} />`}
            </main>
            <footer>Đề tài nghiên cứu: Disentangled-StyleID</footer>`;
    }

    function Root() {
        const { mode, setMode, dark } = useThemeMode();
        const config = {
            algorithm: dark ? theme.darkAlgorithm : theme.defaultAlgorithm,
            token: {
                colorPrimary: dark ? '#1fa899' : '#0b7a70',
                colorInfo: dark ? '#4f9cf0' : '#1d5fbf',
                colorSuccess: dark ? '#62c46a' : '#2b7a33',
                colorWarning: dark ? '#e0a040' : '#965600',
                colorError: dark ? '#f0696c' : '#c4262e',
                colorBgContainer: dark ? '#161f1e' : '#ffffff',
                colorBgElevated: dark ? '#1d2827' : '#ffffff',
                colorBorderSecondary: dark ? '#2b3836' : '#dde5e5',
                borderRadius: 8,
                borderRadiusLG: 12,
                fontSize: 15,
                fontFamily: "'Segoe UI', system-ui, -apple-system, Roboto, sans-serif",
            },
            components: {
                Slider: dark ? { trackBg: '#1fa899', trackHoverBg: '#3cc4b4', handleColor: '#3cc4b4', dotActiveBorderColor: '#3cc4b4' } : {},
                Segmented: dark ? { trackBg: '#0e1514', itemSelectedBg: '#2b3836' } : {},
            },
        };
        return html`<${ConfigProvider} theme=${config}>
            <${Demo} themeMode=${mode} setThemeMode=${setMode} />
        <//>`;
    }

    ReactDOM.createRoot(document.getElementById('root')).render(html`<${Root} />`);
})();
