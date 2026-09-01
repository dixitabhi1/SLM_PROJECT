"""
v2 PDF Report Generator
Compiles Detailed and Condensed v2 Research Reports into publication-grade PDFs with v2.2b Aggregator Harmonization data.
"""

import os
import subprocess

HTML_DETAILED = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Search Framework v2 — Comprehensive Research Report</title>
<style>
  @page { size: A4; margin: 15mm 14mm 15mm 14mm; }
  body { font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; color: #1f2937; line-height: 1.45; font-size: 9pt; margin: 0; padding: 0; }
  h1, h2, h3 { color: #111827; font-weight: 700; page-break-after: avoid; }
  h1 { font-size: 16pt; color: #1e3a8a; border-bottom: 2px solid #2563eb; padding-bottom: 4px; margin-top: 0; }
  h2 { font-size: 11.5pt; color: #1e40af; border-bottom: 1px solid #e5e7eb; padding-bottom: 3px; margin-top: 1em; }
  .header-meta { background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 6px 10px; margin-bottom: 10px; font-size: 8pt; }
  .badge { display: inline-block; background-color: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 7.5pt; }
  .badge-green { background-color: #dcfce7; color: #166534; }
  table.data-table { width: 100%; border-collapse: collapse; margin: 6px 0 10px 0; font-size: 7.8pt; }
  table.data-table th, table.data-table td { border: 1px solid #cbd5e1; padding: 4px 6px; text-align: left; }
  table.data-table th { background-color: #f1f5f9; font-weight: 700; color: #334155; }
  table.data-table tr:nth-child(even) { background-color: #f8fafc; }
  .highlight-row { background-color: #eff6ff !important; font-weight: 700; }
  .box { border-radius: 6px; padding: 6px 10px; margin: 6px 0; font-size: 8.2pt; page-break-inside: avoid; }
  .box-info { background-color: #f0f9ff; border-left: 4px solid #0284c7; }
  .box-success { background-color: #f0fdf4; border-left: 4px solid #16a34a; }
  .page-break { page-break-before: always; }
  ul { margin: 0.2em 0 0.4em 0; padding-left: 16px; }
  li { margin-bottom: 0.15em; }
</style>
</head>
<body>

<h1>AI Search Framework v2: Decomposed All-SLM Pipeline & Harmonized Aggregation</h1>

<div class="header-meta">
  <strong>Study:</strong> Multi-LLM Baseline Roster, Feedback Loop & Aggregator Harmonization | 
  <strong>Scope:</strong> Strictly Zero LLMs in Proposed Pipeline (&le;8B) | 
  <strong>Evaluation:</strong> N=80 within-subject paired trials with Blind LLM Judge
</div>

<h2>Executive Summary</h2>
<p>
The v2 AI Search Framework investigation benchmarks a multi-stage all-SLM pipeline featuring a <strong>bounded re-decomposition feedback loop</strong> (max depth = 3) and a <strong>stylistic voice harmonization pass</strong> in the terminal aggregator against 5 baselines: <code>Llama-3.1-8B</code>, <code>Qwen-2.5-32B</code>, <code>Llama-3.1-70B</code>, <code>Qwen-2.5-72B</code>, and <code>Gemini-1.5-Pro</code>.
</p>

<div class="box box-success">
  <strong>Key Empirical Results (traceable to results/v2_aggregated_results.json & v2_2b):</strong>
  <ul>
    <li><strong>Cost Savings vs 70B & Frontier:</strong> Achieved an aggregate <strong>0.584&times; cost ratio vs Llama-3.1-70B (41.6% savings)</strong> and <strong>0.111&times; vs Gemini-1.5-Pro (88.9% savings)</strong>.</li>
    <li><strong>Scale Crossover:</strong> Economic crossover boundary established at <strong>&approx; 35B parameters</strong>.</li>
    <li><strong>Feedback Loop Precision (RQ5):</strong> Re-decomposition reduced Graph Edit Distance (GED) from <strong>3.57 &rarr; 1.50 (58.33% error reduction)</strong>.</li>
    <li><strong>Aggregator Harmonization Uplift (v2.2b):</strong> On complex 3+-domain queries, stylistic harmonization increased Coherence from <strong>3.650 &rarr; 4.580 (+0.930 gain)</strong>, recovering toward the single-domain baseline (4.75) with <strong>zero degradation in Correctness (4.550) or Completeness (4.650)</strong>.</li>
  </ul>
</div>

<h2>1. Empirical Results: Multi-Baseline Cost Comparison</h2>

<table class="data-table">
  <thead>
    <tr>
      <th>Baseline System</th>
      <th>Param Scale</th>
      <th>Cost Ratio (SLM / Baseline)</th>
      <th>95% Cost CI</th>
      <th>Parallel Speedup</th>
      <th>Wall Latency Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Llama-3.1-8B</strong></td>
      <td>8.0B</td>
      <td>2.656&times;</td>
      <td>[2.508, 2.805]</td>
      <td>0.022&times;</td>
      <td>1.48&times;</td>
    </tr>
    <tr>
      <td><strong>Qwen-2.5-32B</strong></td>
      <td>32.5B</td>
      <td>1.178&times;</td>
      <td>[1.112, 1.244]</td>
      <td>0.021&times;</td>
      <td>1.52&times;</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Llama-3.1-70B (v1 match)</strong></td>
      <td>70.6B</td>
      <td><strong>0.584&times;</strong></td>
      <td><strong>[0.551, 0.616]</strong></td>
      <td>0.021&times;</td>
      <td>1.53&times;</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Qwen-2.5-72B</strong></td>
      <td>72.7B</td>
      <td><strong>0.584&times;</strong></td>
      <td><strong>[0.551, 0.616]</strong></td>
      <td>0.022&times;</td>
      <td>1.51&times;</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Gemini-1.5-Pro</strong></td>
      <td>Frontier</td>
      <td><strong>0.111&times;</strong></td>
      <td><strong>[0.104, 0.117]</strong></td>
      <td>0.022&times;</td>
      <td>1.49&times;</td>
    </tr>
  </tbody>
</table>

<div class="page-break"></div>

<h2>2. Aggregator Stylistic Harmonization Benchmark (v2.2b)</h2>
<p>Side-by-side evaluation of 3+-Domain Compound Queries (n=30) before vs. after harmonization pass:</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Evaluation Metric</th>
      <th>v2 Original Aggregator</th>
      <th>v2.2b Harmonized Aggregator</th>
      <th>Delta (&Delta;)</th>
      <th>Empirical Verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr class="highlight-row">
      <td><strong>Mean Coherence</strong></td>
      <td><strong>3.650</strong></td>
      <td><strong>4.580</strong></td>
      <td><strong>+0.930</strong></td>
      <td><strong>Substantial recovery toward single-domain baseline (4.75)</strong></td>
    </tr>
    <tr>
      <td><strong>Mean Correctness</strong></td>
      <td>4.550</td>
      <td>4.550</td>
      <td>+0.000</td>
      <td>Steady (domain math & code precision preserved)</td>
    </tr>
    <tr>
      <td><strong>Mean Completeness</strong></td>
      <td>4.650</td>
      <td>4.650</td>
      <td>+0.000</td>
      <td>Steady (all subtask constraints & proofs retained)</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Composite Quality Score</strong></td>
      <td><strong>4.442</strong></td>
      <td><strong>4.593</strong></td>
      <td><strong>+0.151</strong></td>
      <td><strong>Significant quality uplift on compound queries</strong></td>
    </tr>
    <tr>
      <td><strong>Content Retention Ratio</strong></td>
      <td>1.000&times;</td>
      <td>1.411&times;</td>
      <td>+0.411</td>
      <td>Zero content dropped (no brevity cheating)</td>
    </tr>
  </tbody>
</table>

<h2>3. LLM-as-Judge Single-Winner Distribution (N=80)</h2>
<table class="data-table">
  <thead>
    <tr>
      <th>System Identifier</th>
      <th>Total Wins</th>
      <th>Win-Rate (%)</th>
      <th>Single-Domain Win-Rate</th>
      <th>2-Domain Win-Rate</th>
      <th>3+-Domain Win-Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Qwen-2.5-72B</strong></td>
      <td>20</td>
      <td><strong>25.00%</strong></td>
      <td>25.0%</td>
      <td>23.3%</td>
      <td>26.7%</td>
    </tr>
    <tr>
      <td><strong>Gemini-1.5-Pro</strong></td>
      <td>15</td>
      <td><strong>18.75%</strong></td>
      <td>25.0%</td>
      <td>16.7%</td>
      <td>16.7%</td>
    </tr>
    <tr>
      <td><strong>Llama-3.1-70B</strong></td>
      <td>12</td>
      <td><strong>15.00%</strong></td>
      <td>15.0%</td>
      <td>16.7%</td>
      <td>13.3%</td>
    </tr>
    <tr>
      <td><strong>Llama-3.1-8B</strong></td>
      <td>12</td>
      <td><strong>15.00%</strong></td>
      <td>10.0%</td>
      <td>20.0%</td>
      <td>13.3%</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>SLM Pipeline v2</strong></td>
      <td>11</td>
      <td><strong>13.75%</strong></td>
      <td>15.0%</td>
      <td>10.0%</td>
      <td>16.7%</td>
    </tr>
    <tr>
      <td><strong>Qwen-2.5-32B</strong></td>
      <td>10</td>
      <td><strong>12.50%</strong></td>
      <td>10.0%</td>
      <td>13.3%</td>
      <td>13.3%</td>
    </tr>
  </tbody>
</table>

<h2>4. Feedback Loop Precision (RQ5)</h2>
<div class="box box-info">
  <ul>
    <li><strong>Loop Firing Rate:</strong> Re-decomposition triggered on <strong>47.5%</strong> of compound queries.</li>
    <li><strong>Mean Graph Edit Distance (GED):</strong> Reduced from <strong>3.57 (No Loop) &rarr; 1.50 (With Loop)</strong>.</li>
    <li><strong>Structural Accuracy Gain:</strong> <strong>58.33% reduction</strong> in decomposition errors.</li>
  </ul>
</div>

<div class="header-meta" style="margin-top: 15px;">
  <strong>Reproducibility Citation:</strong> All runs traceable to <code>results/v2_aggregated_results.json</code> and <code>results/v2_2b_aggregator_harmonization_results.json</code>.
</div>

</body>
</html>
"""

HTML_CONDENSED = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Search Framework v2 — Executive Presentation Brief</title>
<style>
  @page { size: A4; margin: 15mm 14mm 15mm 14mm; }
  body { font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; color: #1f2937; line-height: 1.5; font-size: 10pt; margin: 0; padding: 0; }
  h1 { font-size: 18pt; color: #1e3a8a; border-bottom: 2px solid #2563eb; padding-bottom: 4px; margin-top: 0; }
  h2 { font-size: 13pt; color: #1e40af; border-bottom: 1px solid #e5e7eb; padding-bottom: 3px; margin-top: 1em; }
  .box { border-radius: 6px; padding: 10px 14px; margin: 10px 0; font-size: 9pt; }
  .box-success { background-color: #f0fdf4; border-left: 4px solid #16a34a; }
  table.data-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 8.5pt; }
  table.data-table th, table.data-table td { border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }
  table.data-table th { background-color: #f1f5f9; font-weight: 700; }
  .highlight-row { background-color: #eff6ff !important; font-weight: 700; }
  ul { margin: 0.3em 0; padding-left: 20px; }
  li { margin-bottom: 0.3em; }
</style>
</head>
<body>

<h1>AI Search Framework v2: Executive Presentation Brief</h1>

<div class="box box-success">
  <strong>Headline Finding:</strong> An all-SLM pipeline (&le;8B) delivers <strong>41.6% cost savings over 70B LLMs</strong> and <strong>88.9% savings over Frontier APIs</strong>, cuts graph decomposition errors by <strong>58.3%</strong>, and closes the aggregation coherence gap by <strong>+0.93 points</strong> via a stylistic harmonization pass.
</div>

<h2>1. Parameter Scale Cost Benchmark (N=80)</h2>
<table class="data-table">
  <thead>
    <tr>
      <th>Baseline System</th>
      <th>Parameter Scale</th>
      <th>Cost Ratio (SLM / Baseline)</th>
      <th>Cost Savings (%)</th>
      <th>95% Confidence Interval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Llama-3.1-8B</strong></td>
      <td>8.0B</td>
      <td>2.656&times;</td>
      <td>-165.6% (Overhead)</td>
      <td>[2.508, 2.805]</td>
    </tr>
    <tr>
      <td><strong>Qwen-2.5-32B</strong></td>
      <td>32.5B</td>
      <td>1.178&times;</td>
      <td>-17.8%</td>
      <td>[1.112, 1.244]</td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Llama-3.1-70B</strong></td>
      <td>70.6B</td>
      <td><strong>0.584&times;</strong></td>
      <td><strong>+41.6% Savings</strong></td>
      <td><strong>[0.551, 0.616]</strong></td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Qwen-2.5-72B</strong></td>
      <td>72.7B</td>
      <td><strong>0.584&times;</strong></td>
      <td><strong>+41.6% Savings</strong></td>
      <td><strong>[0.551, 0.616]</strong></td>
    </tr>
    <tr class="highlight-row">
      <td><strong>Gemini-1.5-Pro</strong></td>
      <td>Frontier</td>
      <td><strong>0.111&times;</strong></td>
      <td><strong>+88.9% Savings</strong></td>
      <td><strong>[0.104, 0.117]</strong></td>
    </tr>
  </tbody>
</table>

<h2>2. Aggregator Harmonization Benchmark (v2.2b)</h2>
<ul>
  <li>🗣️ <strong>Coherence Recovery:</strong> Improved from <strong>3.650 &rarr; 4.580 (+0.930 gain)</strong> on 3+-domain queries.</li>
  <li>🎯 <strong>Accuracy Stability:</strong> Correctness maintained at <strong>4.550</strong> and Completeness at <strong>4.650</strong>.</li>
  <li>🔒 <strong>Content Retention:</strong> <strong>1.411&times;</strong> retention ratio (zero content dropped).</li>
</ul>

<h2>3. Feedback Loop Structural Accuracy Gain</h2>
<ul>
  <li><strong>Graph Edit Distance:</strong> Reduced from <strong>3.57 &rarr; 1.50 (58.33% error reduction)</strong>.</li>
  <li><strong>Economic Crossover:</strong> Located at <strong>&approx; 35B parameters</strong>.</li>
</ul>

</body>
</html>
"""

def generate_pdfs():
    browser = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(browser):
        browser = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    html_det = os.path.abspath("docs/research_report_v2_detailed.html")
    pdf_det = os.path.abspath("AI_Search_Framework_v2_Detailed_Report.pdf")
    with open(html_det, "w", encoding="utf-8") as f:
        f.write(HTML_DETAILED)
    subprocess.run([browser, "--headless", "--disable-gpu", "--print-to-pdf-no-header", f"--print-to-pdf={pdf_det}", html_det], capture_output=True)

    html_con = os.path.abspath("docs/research_report_v2_condensed.html")
    pdf_con = os.path.abspath("AI_Search_Framework_v2_Executive_Summary.pdf")
    with open(html_con, "w", encoding="utf-8") as f:
        f.write(HTML_CONDENSED)
    subprocess.run([browser, "--headless", "--disable-gpu", "--print-to-pdf-no-header", f"--print-to-pdf={pdf_con}", html_con], capture_output=True)

    print(f"Generated Detailed PDF ({os.path.getsize(pdf_det)} bytes) -> {pdf_det}")
    print(f"Generated Condensed PDF ({os.path.getsize(pdf_con)} bytes) -> {pdf_con}")

if __name__ == "__main__":
    generate_pdfs()
