// Shared Plotly layout: Arial font, light grid, consistent colors, PNG export.

export const PALETTE = {
  forward: '#2563eb',
  reverse: '#ea580c',
  primary: '#2563eb',
  secondary: '#0891b2',
  accent: '#db2777',
  neutral: '#475569',
}

const AXIS_BASE = {
  showline: true,
  linecolor: '#1e293b',
  linewidth: 1.5,
  mirror: true,
  ticks: 'outside' as const,
  tickcolor: '#1e293b',
  tickwidth: 1.2,
  ticklen: 6,
  showgrid: true,
  gridcolor: '#e2e8f0',
  gridwidth: 1,
  zerolinecolor: '#94a3b8',
  zerolinewidth: 1,
  tickfont: { family: 'Arial, sans-serif', size: 13, color: '#1e293b' },
}

export const AXIS_TITLE_FONT = { family: 'Arial, sans-serif', size: 14, color: '#1e293b' }

export function axisTitle(text: string): { text: string; font: typeof AXIS_TITLE_FONT; standoff: number } {
  return { text, font: AXIS_TITLE_FONT, standoff: 12 }
}

export function baseLayout(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    font: { family: 'Arial, sans-serif', size: 13, color: '#1e293b' },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    margin: { t: 30, r: 40, b: 60, l: 70 },
    xaxis: { ...AXIS_BASE },
    yaxis: { ...AXIS_BASE },
    legend: {
      font: { family: 'Arial, sans-serif', size: 12, color: '#1e293b' },
      bgcolor: 'rgba(255,255,255,0.85)',
      bordercolor: '#cbd5e1',
      borderwidth: 1,
    },
    hovermode: 'closest',
    hoverlabel: {
      font: { family: 'Arial, sans-serif', size: 12 },
      bgcolor: '#ffffff',
      bordercolor: '#cbd5e1',
    },
    ...overrides,
  }
}

export function plotConfig(filename = 'plot'): Record<string, unknown> {
  return {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: {
      format: 'png',
      filename,
      height: 600,
      width: 900,
      scale: 2,
    },
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  }
}

export const LINE = {
  width: 2.5,
}

export const MARKER = {
  size: 7,
  line: { width: 1, color: '#ffffff' },
}
