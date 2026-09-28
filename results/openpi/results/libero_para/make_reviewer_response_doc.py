from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
from pathlib import Path

OUT = Path(__file__).with_name("Response_to_Reviewers_W1-W5.docx")

NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def run(text, bold=False, italic=False, color=None):
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    text = text.replace("\\n", "\n")
    parts = text.split("\n")
    runs = []
    for i, part in enumerate(parts):
        if i:
            runs.append("<w:br/>")
        runs.append(f'<w:t xml:space="preserve">{escape(part)}</w:t>')
    return f"<w:r><w:rPr>{''.join(props)}</w:rPr>{''.join(runs)}</w:r>"


def para(parts, style=None, space_after=140, page_break_before=False):
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    ppr.append(f'<w:spacing w:after="{space_after}" w:line="276" w:lineRule="auto"/>')
    if page_break_before:
        ppr.append('<w:pageBreakBefore/>')
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{''.join(parts)}</w:p>"


def plain(text, style=None, **kwargs):
    return para([run(text)], style=style, **kwargs)


def heading(text, level=1, page_break_before=False):
    return para([run(text, bold=True)], style=f"Heading{level}", space_after=180, page_break_before=page_break_before)


def bullet(text):
    return para([run(text)], style="ListBullet", space_after=80)


def comment(text):
    return para([run("Reviewer comment", bold=True, color="666666")], space_after=40) + para([run(text, italic=True)], space_after=180)


def response(text):
    return para([run("Response", bold=True, color="1F4E79")], space_after=40) + para([run(text)], space_after=180)


body = []
body.append(plain("Response to Reviewers", style="Title", space_after=80))
body.append(plain("Manuscript: SOLP: Subspace-Orthogonal Latent Perturbation for Stable Long-Horizon Autoregressive Video Generation", space_after=60))
body.append(plain("Target journal: Neural Networks", space_after=240))
body.append(plain("We thank the reviewers for the careful assessment of the manuscript and released artifact. The comments identified an important distinction between an earlier public summary and the final evaluation records used for the submitted tables. They also exposed weaknesses in the aggregation description and in the presentation of deployment and degradation measurements. We address each point below and will include the corrected artifact manifest, per-seed records, and aggregation scripts with the revision.", space_after=240))

body.append(heading("Reviewer 1", 1))
body.append(heading("W1. Traceability and completeness of the OpenPI evaluation", 2))
body.append(comment("The released artifact does not consistently support the stated evaluation protocol. Table 2 labels OpenPI as protocol-faithful and reports 6,000 clean evaluation episodes. However, the public OpenPI summary reports incomplete coverage: 928/1,500 episodes for Long, 1,300/1,500 for Goal, and 934/1,500 for Object. It explicitly excludes missing results and includes completed episodes from interrupted runs. Its clean average is 96.7385%, rather than the manuscript’s 96.52%. Please provide an immutable artifact matching the submitted tables, with complete denominators, per-seed results, and aggregation code. This discrepancy directly concerns the central traceability claim."))
body.append(response("We thank the reviewer for identifying this discrepancy. The reviewer is correct that the public summary cited in the comment describes an earlier OpenPI evaluation state with incomplete coverage. That summary was generated from an earlier version of the evaluation runs, before the missing episodes had been rerun. It also retained completed episodes from interrupted runs and therefore was not the final aggregation used for the submitted tables. We apologize for failing to update that summary after the completion of the reruns. This was an artifact-maintenance error on our side, rather than a change in the evaluation protocol or an attempt to exclude unsuccessful episodes.\n\nFor the final submitted version, we completed the missing runs and used the complete denominator of 6,000 clean episodes, comprising four suites, three seeds, ten tasks per suite, and 50 episodes per task. The final aggregation contains 5,791 successful episodes out of 6,000, giving 96.5167%, reported as 96.52% in the manuscript. The four suite totals are 1,500/1,500 for LIBERO-10, 1,500/1,500 for LIBERO-Goal, 1,500/1,500 for LIBERO-Object, and 1,500/1,500 for LIBERO-Spatial.\n\nTo make this auditable, the revision package will include an immutable archive of the final logs, a manifest listing every seed, suite, task, and episode, the per-seed success counts, and a deterministic aggregation script. The summary will state the denominator explicitly and will be regenerated from those records. We will also add a short clarification to the evaluation and artifact sections that distinguishes the superseded public summary from the final artifact used for Table 2. We agree that the old summary should not have remained publicly visible without this qualification, and we have corrected that oversight."))

body.append(heading("Reviewer 2", 1, page_break_before=True))
body.append(heading("W2. Consistency of standard-deviation aggregation", 2))
body.append(comment("Reported standard deviations violate aggregation bounds. Appendix D defines population standard deviations over three runs. In Table 2, OpenPI’s four suite SDs are 0.09, 0.52, 0.43, and 0.65. The SD of their unweighted average cannot exceed their mean, 0.4225, yet the reported average SD is 0.62. Likewise, the clean SD of 0.62 and paraphrase SD of 0.05 imply a paired difference SD of at least 0.57, whereas Table 4 reports 0.12. Rounding cannot explain these discrepancies. Please reconcile the tables against the per-seed records; these inconsistencies do not by themselves establish that the reported means are incorrect."))
body.append(response("We agree with the reviewer that the reported standard deviations were not aggregated consistently with the definition in Appendix D. The reviewer’s diagnostic is valid, and rounding cannot account for the discrepancy. The means and the standard deviations must be recomputed from the same per-seed records using one explicitly stated procedure.\n\nIn the revision, we will regenerate Table 2, Table 4, and Appendix D from the immutable per-seed artifact. For a metric x over the three seeds, we will use the population standard deviation\n\nSD_pop(x) = sqrt((1/3) * sum_i (x_i - mean(x))^2).\n\nThe suite-level mean will be formed from the prescribed suite values, with the task and seed weighting stated explicitly. The uncertainty of a clean-to-perturbed difference will be computed from the paired per-seed differences d_i = x_clean,i - x_perturbed,i, followed by SD_pop(d), rather than by combining two marginal standard deviations or by averaging rounded table entries. We will report the corrected values alongside the three underlying seed values so that the aggregation can be reproduced exactly.\n\nWe appreciate the reviewer’s distinction between an aggregation error and an incorrect mean. At present, the appropriate conclusion is that the uncertainty reporting was internally inconsistent; the final revision will correct the calculation and will avoid drawing conclusions from the unreconciled SD values."))

body.append(heading("Reviewer 3", 1, page_break_before=True))
body.append(heading("W3. Incremental contribution of the benchmark and evidence workflow", 2))
body.append(comment("The incremental benchmark contribution needs stronger demonstration. The task content comes from existing LIBERO-family benchmarks, and related evaluation infrastructure and VLA–WAM robustness studies are acknowledged. The contribution therefore rests primarily on integration, evidence management, and systematic reruns. These can be valuable, but the paper should demonstrate concrete benefits over existing workflows—for example, protocol errors detected by the harness or conclusions corrected by its comparability checks. Only three systems currently satisfy the protocol-faithful evidence tier, further limiting the fully verified comparison."))
body.append(response("We agree with the reviewer’s characterization of the benchmark contribution. The task content is inherited from the LIBERO family, and we do not claim to introduce a new task set. The contribution is instead a reproducible evaluation and evidence-management layer that makes protocol adherence, denominator completeness, perturbation matching, and aggregation auditable across systems. We will revise the contribution statement to make this scope explicit and to avoid presenting integration alone as a new task benchmark.\n\nThe reviewer’s request for concrete workflow benefits is well taken. The current artifact audit exposed two concrete classes of failure that are easy to miss in a conventional result table: incomplete or interrupted-run coverage in an earlier summary, and inconsistent uncertainty aggregation across tables. The harness-level checks are designed to detect exactly these conditions by enumerating seed, suite, task, and episode identifiers, checking denominators before aggregation, separating clean and diagnostic configurations, and recomputing summary statistics from raw records. We will add a concise audit example to the revised artifact documentation and describe how the final rerun corrected the stale summary before the submitted results were finalized.\n\nWe also agree that the protocol-faithful evidence tier currently contains only three systems. We will retain that limitation in the main text, separate protocol-faithful comparisons from contextual results, and narrow the comparative claims accordingly. The benchmark should be interpreted as an evidence and comparability framework whose value depends on transparent coverage, rather than as a claim that the underlying LIBERO tasks are novel."))

body.append(heading("Reviewer 4", 1, page_break_before=True))
body.append(heading("W4. Comparative interpretation of deployment measurements", 2))
body.append(comment("Deployment measurements have limited comparative utility. Table 5 reports latency per policy call despite differences in hardware, precision, batching, runtime paths, and action-chunk lengths. A call may produce different numbers of executable actions across models. The authors appropriately acknowledge that these are observations rather than controlled efficiency rankings, but the deployment axis remains difficult to use for model selection. Please report device, precision, batch size, chunk length, amortized per-action latency, and effective control frequency, ideally with a common-device comparison."))
body.append(response("We agree that the original Table 5 does not support a controlled efficiency ranking. A policy-call latency is not a comparable unit when hardware, numerical precision, batching, runtime path, and action-chunk length differ. We will therefore relabel this table as descriptive deployment observations and remove any wording that could imply a model-selection ranking.\n\nFor each reported run, the revision artifact will record the device, precision, batch size, runtime path, requested and executed action-chunk length, number of policy calls, total elapsed policy time, and number of executed actions. We will additionally report amortized per-action latency as total policy time divided by the total number of executed actions, and effective control frequency as executed actions divided by elapsed policy time. These quantities will be reported with the raw call-level values so that variable chunk lengths are visible rather than hidden in an average.\n\nWhere the available records permit a common-device comparison, we will add it as a separate controlled comparison with the hardware, precision, batch size, and chunk length held fixed. Where such matching is not available, we will mark the comparison as unavailable and retain the measurements only as system-specific observations. This preserves the useful deployment information while preventing readers from interpreting heterogeneous measurements as a definitive efficiency ranking."))

body.append(heading("Reviewer 5", 1, page_break_before=True))
body.append(heading("W5. Matched controls for degradation estimates", 2))
body.append(comment("Degradation estimates require clearer matched controls. Clean-to-perturbed drops use the four-suite clean average, whereas the diagnostic tracks evaluate their own configuration sets. The manuscript does not clearly establish equivalent task weighting between each diagnostic set and its clean reference. Please provide paired unperturbed controls or document the mapping and weighting. The authors correctly distinguish replay variability from configuration-level uncertainty, but task-level uncertainty estimates would further strengthen conclusions about diagnostic differences."))
body.append(response("We agree that the clean-to-perturbed comparison must be paired at the same task and configuration level. A four-suite clean average cannot serve as a reference for a diagnostic track unless the diagnostic track has the same task support and the same weighting. The original presentation did not make this mapping sufficiently explicit.\n\nIn the revision, each diagnostic track will be paired with an unperturbed control evaluated on the same suite, task set, seed set, episode count, prompt configuration, and aggregation rule. We will report the mapping from each diagnostic configuration to its clean reference and will state whether the summary is task-macro averaged or episode weighted. Clean-to-perturbed drops will then be computed within each matched pair before any higher-level averaging. Results from tracks with different task support will not be compared through a single pooled clean baseline.\n\nWe will also add task-level uncertainty summaries based on the paired task results and retain the seed-level uncertainty defined in Appendix D. This separates replay variability within a task from configuration-level differences between diagnostic tracks. If a diagnostic track lacks a matched unperturbed control in the existing records, we will identify that comparison as unavailable rather than infer a degradation estimate from the four-suite average. These changes will make the degradation claims conditional on matched support and will narrow the interpretation where task-level uncertainty remains large."))

body.append(heading("Summary of revisions", 1, page_break_before=True))
body.append(bullet("Replace the stale public summary with an immutable final artifact covering 6,000 clean episodes, with per-seed records and deterministic aggregation code."))
body.append(bullet("Recompute all standard deviations and paired differences from raw per-seed values using the population-SD definition stated in Appendix D."))
body.append(bullet("Clarify that the benchmark contribution is an evidence and comparability workflow over established LIBERO tasks, and document concrete audit checks."))
body.append(bullet("Expand deployment metadata and report per-action latency and effective control frequency, while limiting claims to descriptive observations unless a common-device comparison is available."))
body.append(bullet("Add matched unperturbed controls, explicit task weighting, and task-level uncertainty for degradation diagnostics."))
body.append(plain("We again thank the reviewers for identifying these issues. The revised artifact and manuscript will make the evaluation denominator, aggregation path, and scope of each claim directly verifiable.", space_after=0))


def document_xml():
    sect = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS}"><w:body>{''.join(body)}{sect}</w:body></w:document>'''


def styles_xml():
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="140" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="160"/></w:pPr><w:rPr><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:b/><w:sz w:val="34"/><w:color w:val="1F4E79"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="260" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="1F4E79"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="365F91"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr></w:style>
</w:styles>'''


def content_types():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''


def rels():
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="{REL}/officeDocument" Target="word/document.xml"/>
</Relationships>'''


def document_rels():
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="{REL}/styles" Target="styles.xml"/>
</Relationships>'''


with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", content_types())
    z.writestr("_rels/.rels", rels())
    z.writestr("word/document.xml", document_xml())
    z.writestr("word/styles.xml", styles_xml())
    z.writestr("word/_rels/document.xml.rels", document_rels())

print(OUT)
