declare module 'plotly.js-dist-min' {
  const Plotly: {
    newPlot(
      root: string | HTMLElement,
      data: Array<Record<string, unknown>>,
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ): Promise<void>
    purge(root: string | HTMLElement): void
  }
  export default Plotly
}
