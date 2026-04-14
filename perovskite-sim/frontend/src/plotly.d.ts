// The frontend only uses `scatter` traces + log/linear axes, which are all
// included in the basic bundle. Swapping to plotly.js-basic-dist-min drops
// the production bundle from ~4.8 MB to ~1.3 MB.
declare module 'plotly.js-basic-dist-min' {
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
