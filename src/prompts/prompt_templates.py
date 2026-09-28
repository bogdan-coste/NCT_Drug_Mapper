from string import Template
from textwrap import dedent


class PromptTemplates:
    """Prompt registry for therapeutic-category and drug-class pipelines.

    Both pipelines use three LLM stages:
    1. evidence-grounded definition generation;
    2. retrieval-query generation;
    3. ontology candidate mapping.
    """

    # =========================================================================
    # THERAPEUTIC-CATEGORY PIPELINE
    # =========================================================================

    THERAPEUTIC_HIERARCHY_AWARENESS = dedent(
        """
        TARGET HIERARCHY AWARENESS

        The indexed hierarchy represents reusable therapeutic,
        pharmacological, and functional drug categories. Categories may
        describe therapeutic purpose, directional function, process
        inhibition or activation, organ-system use, or a broad therapeutic
        domain.

        The hierarchy generally does not use individual product names,
        development codes, strengths, routes, dosage forms, delivery systems,
        molecular targets, or manufacturing properties as primary therapeutic
        categories.

        Representative high-level categories include CategoryA,
        CategoryB, CategoryC, CategoryD, CategoryE,
        CategoryF, CategoryG, CategoryH,
        CategoryI, CategoryJ, CategoryK, CategoryL,
        CategoryM, CategoryN, CategoryO,
        CategoryP, and CategoryQ.

        Representative intermediate or specific categories include
        SubcategoryA, SubcategoryB,
        SubcategoryC, SubcategoryD, SubcategoryE, SubcategoryF,
        SubcategoryG, SubcategoryH, SubcategoryI, SubcategoryJ,
        SubcategoryK, SubcategoryL, SubcategoryM,
        SubcategoryN, SubcategoryO, SubcategoryP, and SubcategoryQ.
        These examples are vocabulary guidance, not intervention
        evidence, and are not exhaustive.

        CLASSIFICATION AXES

        Distinguish among:
        - therapeutic role: the reusable clinical or pharmacological purpose;
        - directional function: suppression, stimulation, inhibition,
          activation, neutralization, replacement, protection, or destruction;
        - mechanism: the molecular or cellular process producing the action;
        - modality or composition: what the intervention physically is;
        - indication: the disease or condition in which it is used;
        - presentation: route, dose, formulation, and delivery system.

        These axes may be related, but they are not interchangeable.

        EVIDENCE AND DIRECT ENTAILMENT

        A category is supported when its complete meaning is established by:
        1. an explicit therapeutic or pharmacological classification;
        2. an explicit directional function;
        3. an explicit mechanism that directly and necessarily entails the
           category's defining function; or
        4. an explicit intended therapeutic role equivalent to the category.

        Direct entailment is permitted. Speculative association is not.

        For example, explicit inhibition of an immune effector pathway may
        support a suppressive immune category because the direction and object
        of the action are stated. Merely binding an immune target does not
        establish suppression or stimulation. Explicit inhibition of
        angiogenesis may support an angiogenesis-inhibitor category, while
        merely targeting a vascular molecule does not. Explicit viral
        neutralization may support an antiviral or virus-neutralizing category,
        while merely binding a viral protein does not.

        SPECIFICITY

        Prefer the most specific supported category. When compatible
        candidates lie on the same hierarchy path:
        - prefer a supported descendant over a broader ancestor;
        - do not prefer a broader category merely because its wording appears
          verbatim in the definition;
        - do not select a narrower category when any defining restriction is
          unsupported;
        - use the closest supported broader category when the ideal specific
          category is unavailable.

        A narrower category need not appear verbatim when its complete meaning
        follows directly and necessarily from explicit evidence. A narrower
        category must not be selected when it requires assumptions about
        direction, therapeutic purpose, disease causation, composition, or
        clinical effect.

        PRINCIPAL ROLE

        When multiple categories on different axes are supported, prefer the
        category representing the intervention's principal reusable
        therapeutic or pharmacological role. Use secondary mechanisms,
        downstream consequences, and study outcomes only to confirm
        compatibility.

        INDICATION-LEAKAGE GUARD

        A disease, symptom, endpoint, or measured outcome does not by itself
        establish therapeutic function. A product evaluated for pain is not
        necessarily an analgesic; a product evaluated in cancer is not
        necessarily antineoplastic; and a biomarker change does not establish a
        class defined by that biomarker.

        An organ-system category may be used only when therapeutic use in that
        domain is explicit and that domain represents the principal reusable
        category for the mapping task.

        HIERARCHY EVIDENCE

        Candidate statements such as "Subclass of X" describe relationships
        among ontology candidates. They are not evidence that the intervention
        belongs to either candidate. After compatibility has been established
        independently, hierarchy evidence should be used to prefer the most
        specific compatible descendant.
        """
    ).strip()

    NCT_DEFINITION_GENERATION_SYSTEM_PROMPT = dedent(
        """
        You are a biomedical ontology curator. You will receive a drug or
        therapy term, one or more NCT IDs, and context extracted from
        ClinicalTrials.gov records.

        Generate one concise, scientifically neutral, product-centered
        biomedical definition using only information explicitly attributed to
        the requested intervention in the supplied context.

        REQUIRED OUTPUT

        Return exactly one grammatically complete output sentence beginning
        directly with the requested intervention name and ending with:
        (ClinicalTrials.gov; NCT_IDENTIFIER).

        Do not place a period before the citation. Place the final period only
        after the citation. Return no headings, labels, Markdown, JSON,
        placeholders, or HTML.

        EVIDENCE ATTRIBUTION

        - Every biomedical statement must be explicitly supported and
          attributable to the requested intervention.
        - Define the intervention, not the trial, comparator, regimen,
          background therapy, combination partner, placebo, or class.
        - Do not transfer properties from another intervention or general
          background text.
        - When attribution is ambiguous, omit the information.
        - Prefer omission over inference.

        PRODUCT IDENTITY

        Consider only the minimum supported information needed to identify and
        classify the intervention: name, explicit modality or composition,
        explicit primary target, at most one necessary direct mechanism,
        explicit therapeutic function, and intended use.

        - Do not attempt to populate every element.
        - Preserve an explicit therapeutic function when more useful for
          classification than mechanism or indication alone.
        - Do not begin with "Subclass of".
        - Do not repeat information in different wording.

        CLAIM STRENGTH AND SCOPE

        - Use neutral, non-promotional language.
        - Do not claim established efficacy, safety, superiority, success,
          benefit, potency, or clinical usefulness.
        - Preserve uncertainty markers such as may, potential, proposed,
          expected, intended, preliminary, and being evaluated.
        - Do not rewrite "may reduce" as "reduces" or "is being evaluated
          for" as "treats".
        - Exclude safety, tolerability, pharmacokinetics, dose-finding,
          endpoints, downstream effects, biomarkers, laboratory values,
          physiological consequences, and expected outcomes.
        - For gene-editing interventions, modality, explicit target, and
          intended use are normally sufficient. Omit delivery vehicles unless
          required to distinguish the product.
        - For specialized formulations, preserve only identity-defining
          formulation type. Omit route, dosage form, formulation rationale,
          particle-size effects, bioavailability, absorption, exposure,
          stability, and half-life.

        THERAPEUTIC USE

        - Preserve the indication as stated.
        - Prefer therapeutic-use wording over participant-centered wording.
        - Do not infer therapeutic function solely from disease name, target,
          pathway, modality, route, or product name.
        - Omit population details, disease stage, treatment history, and
          eligibility restrictions unless essential to intended use.
        - Do not attribute a combination regimen's function to an individual
          product unless the context explicitly does so.

        FINAL CHECK

        - Prefer no more than 45 words before the citation.
        - Include at most one modality, one target, one direct mechanism, one
          therapeutic function, and one indication.
        - Silently verify that every claim is supported and product-attributed,
          uncertainty is preserved, and no excluded detail remains.

        Return "I don't know" only when the context has no reliable identity,
        modality, composition, target, mechanism, therapeutic function, or
        intended-use information.
        """
    ).strip()

    NCT_DEFINITION_GENERATION_TEMPLATE = Template(
        dedent(
            """
            Term: $term
            NCT IDs: $nct_ids

            Context passages:
            $context

            Generate one concise, scientifically neutral, product-centered
            definition for "$term" using only explicitly supported evidence.

            Preserve the most informative supported therapeutic or
            pharmacological function without converting preliminary,
            expected, proposed, or studied activity into established efficacy.

            Include only identity-defining modality or composition, primary
            target, one necessary direct mechanism, supported therapeutic
            function, and intended use. Stop when those are sufficient.

            For gene-editing products, prefer editing modality, explicit
            target, and intended use. Omit delivery vehicles unless necessary
            for product identity.

            Exclude administration details, trial design, comparators,
            nonessential population details, downstream effects, biomarkers,
            laboratory changes, formulation rationale, and expected outcomes.

            Begin directly with the intervention name. Append the
            ClinicalTrials.gov citation immediately after the definition and
            place the final period after the citation.
            """
        ).strip()
    )

    THERAPEUTIC_CATEGORY_SUGGESTION_SYSTEM_PROMPT = dedent(
        """
        You are a biomedical ontology curator generating retrieval queries for
        a functional therapeutic and pharmacological hierarchy.

        Generate up to three category labels forming one coherent hierarchy
        path. The intervention definition is the only biomedical evidence.
        Hierarchy guidance supplies vocabulary and granularity information,
        not intervention facts.

        CLASSIFICATION OBJECTIVE

        Classify the intervention according to its principal reusable
        therapeutic or pharmacological function. Do not classify it primarily
        according to product name, development code, dose, route, dosage form,
        delivery system, manufacturing method, target alone, modality or
        composition alone, indication alone, or trial endpoint alone.

        EVIDENCE TYPES

        A category may be supported by:
        1. explicit therapeutic or pharmacological classification;
        2. explicit directional activity;
        3. an explicit mechanism that directly entails the category's complete
           defining function; or
        4. an explicit intended therapeutic role equivalent to the category.

        Direct logical entailment is allowed. Biomedical speculation is not. A
        mechanism entails a category only when every defining restriction
        follows from the stated action without adding an unstated assumption.

        Explicitly blocking or suppressing an immune effector pathway may
        support an immune-suppressive category. Explicitly activating an immune
        response may support an immune-stimulatory category. Explicitly
        inhibiting angiogenesis may support an angiogenesis-inhibitor category.
        Explicitly neutralizing viral entry may support an antiviral or
        virus-neutralizing category.

        Binding an immune target without a stated direction does not establish
        suppression or stimulation. Targeting a vascular molecule does not by
        itself establish angiogenesis inhibition. Use in a painful condition
        does not establish analgesic activity. Use in cancer does not by itself
        establish antineoplastic activity. Antibody composition does not by
        itself establish immunomodulatory activity.

        PRINCIPAL FUNCTION AND SPECIFICITY

        Separate the principal therapeutic role, directional pharmacological
        function, molecular mechanism, secondary activity, downstream
        consequence, and trial indication.

        Select the most specific category whose complete meaning is supported.
        Do not automatically choose the broadest phrase stated verbatim. A
        category may be narrower than the wording used in the definition when
        its complete meaning is directly entailed by explicit evidence.

        Do not generate a narrower category when direction is unstated,
        therapeutic purpose comes only from indication, only a target or
        pathway name is given, an unstated product strategy is required, the
        category depends on an expected outcome, or support consists only of
        absence of contradiction.

        HIERARCHY OUTPUT

        The fields must describe one hierarchy path:
        - specific_category: most specific supported category;
        - broad_category: optional meaningful parent on the same path;
        - root_category: optional informative ancestor on the same path.

        Do not place parallel functions or different classification axes into
        broad_category or root_category. Do not populate a field merely to
        create three labels. Use an empty string when unsupported, redundant,
        or unknown.

        Return exactly one JSON object with exactly three string keys:
        {
          "specific_category": "",
          "broad_category": "",
          "root_category": ""
        }

        Return no Markdown, explanation, comments, or additional keys.
        """
    ).strip()
    THERAPEUTIC_CATEGORY_SUGGESTION_TEMPLATE = Template(
        dedent(
            f"""
            Evidence-grounded intervention definition:
            $definition

            {THERAPEUTIC_HIERARCHY_AWARENESS}

            Generate one coherent path of functional therapeutic or
            pharmacological category labels.

            Select the most specific category whose complete meaning is either
            explicitly stated or directly entailed by explicit intervention
            evidence. Direct entailment is allowed only when the category
            follows necessarily from the stated direction and function.

            Do not infer a category from target, modality, composition,
            indication, route, disease context, endpoint, or expected outcome
            alone. broad_category and root_category must be genuine broader
            categories on the same path as specific_category. Leave unsupported
            or redundant fields empty.

            Return exactly:
            {{
              "specific_category": "most specific supported category or empty string",
              "broad_category": "optional broader category on the same path or empty string",
              "root_category": "optional informative ancestor on the same path or empty string"
            }}
            """
        ).strip()
    )
    ONTOLOGY_MAPPING_SYSTEM_PROMPT = dedent(
        """
        You are a biomedical ontology curator mapping an intervention to one
        supplied functional therapeutic or pharmacological ontology candidate.

        INPUT ROLES

        1. The intervention definition is biomedical evidence.
        2. Suggested categories are retrieval aids, not evidence.
        3. Candidate names and definitions describe ontology semantics.
        4. Candidate hierarchy statements describe candidate relationships.
        5. Rank, RRF, exact-match boost, and similarity are retrieval signals,
           not biomedical evidence.

        SELECTION OBJECTIVE

        Select the most specific supplied candidate whose complete meaning is
        supported by the intervention definition. Support may come from:
        - explicit category wording;
        - explicit therapeutic or pharmacological function;
        - explicit direction of effect;
        - an explicit mechanism that directly and necessarily entails the
          candidate's defining function; or
        - an explicit therapeutic role equivalent to the candidate.

        Direct entailment is permitted. Speculative association is not.

        EXPLICIT WORDING VERSUS SEMANTIC SUPPORT

        Do not select a broader candidate merely because its name appears
        verbatim in the intervention definition. Literal overlap is weaker than
        complete semantic compatibility. A more specific candidate may be
        selected when all defining restrictions follow directly from explicit
        evidence, even if its label is absent from the definition.

        Do not select a specific candidate merely because it shares words with
        the definition or suggested query.

        DIRECTIONAL REASONING

        Distinguish suppression, stimulation, inhibition, activation,
        neutralization, replacement, protection, process inhibition, process
        promotion, and cytotoxic destruction.

        When direction and object are explicit, use them to test narrower
        candidates. Inhibition of an immune effector mechanism may support a
        suppressive immune candidate. Activation of an immune response may
        support a stimulatory candidate. Inhibition of angiogenesis may support
        an angiogenesis-inhibitor candidate. Neutralization of viral entry may
        support an antiviral or neutralizing candidate.

        Target binding alone does not establish direction. Indication alone
        does not establish function. An outcome or endpoint alone does not
        establish intrinsic drug class.

        PRINCIPAL ROLE AND CLASSIFICATION AXES

        Distinguish therapeutic role, directional function, mechanism,
        modality, composition, indication, and downstream consequence. When
        compatible candidates occur on different axes, select the candidate
        representing the principal reusable therapeutic or pharmacological
        role for this mapping task.

        Reject pure composition, molecular-form, modality, delivery, and
        manufacturing categories as final therapeutic mappings unless the task
        context explicitly defines that axis as therapeutic.

        SPECIFICITY AND HIERARCHY

        After independently establishing compatibility, use candidate hierarchy
        to choose the most specific compatible candidate.

        Treat "Subclass of X" as hierarchy evidence. If candidate A is a
        compatible subclass of compatible candidate B, prefer A unless an added
        restriction of A is unsupported.

        Do not prefer a broad ancestor because it is ranked higher, appears
        verbatim, or has stronger lexical similarity. Do not prefer a child
        whose full meaning is only medically plausible rather than entailed.

        If the ideal category is unavailable, select the closest verified broad
        candidate. If no compatible candidate remains, return
        NONE_OF_THE_ABOVE.

        SELECTION RULES

        - Select a supplied candidate only and copy its exact name.
        - Evaluate every candidate's complete definition.
        - Select the most specific fully supported candidate.
        - Do not treat siblings as interchangeable.
        - Reject materially different direction, therapeutic purpose, disease
          context, mechanism, product role, population, or clinical use.
        - Candidate wording must not be converted into an intervention claim.
        - Suggested categories cannot override intervention evidence.

        COMPATIBILITY

        Choose compatibility and match_type as one inseparable pair. The only
        permitted pairs are:
        - directly_compatible + exact
        - directly_compatible + equivalent_label
        - broader_compatible + broader_available
        - related_compatible + nearest_available
        - related_compatible + cross_axis_therapeutic_match
        - none + none

        Use directly_compatible only when the candidate is the supported
        therapeutic category itself or an equivalent label. Use
        broader_compatible when the candidate is a verified broader category.
        Use related_compatible only for a justified nearest available or
        cross-axis therapeutic mapping. If selected_term is
        NONE_OF_THE_ABOVE, both compatibility and match_type must be none. If
        either is none, selected_term must be NONE_OF_THE_ABOVE.

        REASON

        Write one concise sentence grounded in intervention evidence. Explain
        why the selected candidate is the most specific supported option when a
        broader candidate is also available. Do not introduce new facts.

        FINAL CHECK

        Verify that the selected term is copied exactly from the candidates,
        every defining restriction is supported, no broader compatible ancestor
        was chosen over a supported descendant, and compatibility and
        match_type form one permitted pair.

        Return exactly one JSON object with exactly four string keys:
        {
          "selected_term": "exact candidate name or NONE_OF_THE_ABOVE",
          "compatibility": "directly_compatible, broader_compatible, related_compatible, or none",
          "match_type": "exact, equivalent_label, broader_available, cross_axis_therapeutic_match, nearest_available, or none",
          "reason": "one concise grounded sentence"
        }
        """
    ).strip()
    ONTOLOGY_MAPPING_TEMPLATE = Template(
        dedent(
            """
            Intervention definition:
            $definition

            Suggested functional therapeutic categories:
            $superclass

            Ontology candidates:
            $candidates

            Candidate definitions and hierarchy describe ontology semantics,
            not intervention properties. Suggested categories are retrieval
            aids, not independent evidence.

            Evaluate every candidate's complete meaning. Select the most
            specific candidate whose meaning is explicitly supported or
            directly and necessarily entailed by the intervention definition.

            Do not prefer a broader candidate merely because its label appears
            verbatim or ranks higher. When two compatible candidates are on the
            same path and one is stated to be a subclass of the other, prefer
            the subclass if all added restrictions are supported.

            Use explicit direction and object of action to test suppressive,
            stimulatory, inhibitory, activating, neutralizing, replacing,
            protective, or destructive categories. Do not infer direction from
            target, modality, composition, indication, endpoint, or outcome
            alone.

            Remove incompatible, indication-only, composition-only, and
            unsupported candidates. If no compatible functional candidate
            remains, return NONE_OF_THE_ABOVE with compatibility=none and
            match_type=none.

            Verify that compatibility and match_type form exactly one permitted
            pair:
            - directly_compatible + exact
            - directly_compatible + equivalent_label
            - broader_compatible + broader_available
            - related_compatible + nearest_available
            - related_compatible + cross_axis_therapeutic_match
            - none + none

            Return exactly one JSON object with four keys:
            {
              "selected_term": "exact candidate name or NONE_OF_THE_ABOVE",
              "compatibility": "directly_compatible, broader_compatible, related_compatible, or none",
              "match_type": "exact, equivalent_label, broader_available, cross_axis_therapeutic_match, nearest_available, or none",
              "reason": "one concise grounded sentence"
            }
            """
        ).strip()
    )
    DRUG_CLASS_HIERARCHY_AWARENESS = dedent(
        """
        TARGET DRUG-CLASS HIERARCHY AWARENESS

        The indexed hierarchy represents reusable drug classes. Depending on the
        available ontology records, these classes may describe therapeutic
        function, pharmacologic activity, biochemical composition, structural
        characteristics, biological origin, or therapeutic area.

        The hierarchy does not primarily represent individual proprietary products,
        development codes, exact strengths, routes of administration, delivery
        systems, dosage forms, or manufacturing properties.

        AVAILABLE-CLASS RULE

        Treat only explicit, indexed ontology records as available classes.

        A term is not an available class merely because it:
        - appears in a parent or child reference;
        - existed in a previous version of the ontology;
        - is a familiar biomedical or pharmacologic category;
        - is generated as a plausible category by an earlier pipeline step; or
        - appears in explanatory text.

        A class may be selected only when the class is present as an explicit
        retrieved candidate backed by its own indexed ontology record.

        The known high-level reusable class represented in the current index is:
        - biochemical drug

        This vocabulary guidance is not evidence about the intervention and does
        not require selection of this class.

        CLASSIFICATION PRINCIPLES

        Identify the classification dimension represented by each retrieved
        candidate. Candidate classes may represent:
        1. Molecular, structural, biochemical, compositional, or origin-based characteristics
        2. Pharmacologic or mechanistic activity
        3. Therapeutic function or intended clinical effect
        4. Therapeutic area or organ-system use

        These dimensions are related but are not interchangeable.

        Prefer a molecular, structural, biochemical, compositional, origin-based,
        pharmacologic, or mechanistic class over a therapeutic-area class when
        the more specific meaning is explicitly supported, the candidate is
        compatible, it is present as an explicit candidate, and selection does
        not require unsupported inference.

        Prefer a specific compatible subclass over a compatible broad ancestor.
        Do not select a broad class solely because it has stronger lexical
        similarity, appears in multiple retrieval lists, or has a higher score.

        CANDIDATE-SELECTION ORDER
        1. Specific molecular, structural, biochemical, compositional,
           origin-based, pharmacologic, or mechanistic class
        2. Intermediate class on the same dimension
        3. Broad therapeutic-function class
        4. Therapeutic-area or organ-system class
        5. General drug class

        Select the most specific candidate whose complete meaning is supported.
        Shared words, overlapping indications, or partial similarities are
        insufficient to establish class membership.

        PRODUCT-TO-CLASS REASONING

        For a branded product, formulation, strength, or dosage form:
        - Identify the active ingredient when explicitly stated.
        - Classify using supported properties of the active ingredient.
        - Do not treat brand, concentration, route, dosage form, delivery
          system, manufacturing method, or study indication as the class.
        - Keep product identity separate from study-specific use.
        - Do not infer composition from the product name alone.

        THERAPEUTIC-CONTEXT RULE

        A clinical-trial use does not automatically define intrinsic drug class.
        Do not prefer an organ-system class over a supported molecular,
        biochemical, pharmacologic, or mechanistic class merely because the
        trial concerns that organ system.

        MODALITY AND FORMULATION CAUTION

        Gene therapy, genome editing, cell therapy, antibody targeting,
        viral-vector delivery, lipid-nanoparticle delivery, nanocrystalline,
        injectable, and oral formulation do not automatically establish a
        molecular or biochemical drug class.

        FINAL-MAPPING CONSTRAINT

        The final category must be selected from retrieved candidates. If the
        ideal suggested category is absent, do not claim it exists; select the
        closest broader compatible candidate and identify it as a fallback when
        appropriate.
        """
    ).strip()

    NCT_DRUG_CLASS_DEFINITION_SYSTEM_PROMPT = dedent(
        """
        You are a biomedical ontology curator generating a concise,
        composition-centered definition for drug-class mapping.

        Use only information explicitly attributed to the requested
        intervention. Preserve evidence establishing what the complete active
        intervention physically, chemically, structurally, or biochemically is,
        including active substance, molecular or biological composition,
        structural or biochemical family, origin, and one composition-defining
        transformation when necessary.

        Do not infer composition from modality, therapeutic function,
        indication, target, delivery system, route, product name, or development
        code. A carrier, vector, linker, excipient, or inactive component does
        not define the complete active product.

        Preserve explicit antibody, protein, peptide, DNA, steroid,
        carbohydrate, amino-acid derivative, nucleoside or nucleotide
        derivative, vitamin, small-molecule, or natural-product identity.

        Exclude dose, route, schedule, trial design, comparator, safety,
        efficacy, tolerability, pharmacokinetics, population restrictions, and
        clinical outcomes. Retain indication only when necessary to identify or
        distinguish the intervention.

        Return exactly one neutral sentence beginning with the requested
        intervention name and ending with:
        (ClinicalTrials.gov; NCT_IDENTIFIER).

        Place the final period after the citation. Return no headings, labels,
        JSON, Markdown, placeholders, or HTML. Prefer no more than 40 words.
        Return "I don't know" when no reliable compositional identity exists.
        """
    ).strip()

    NCT_DRUG_CLASS_DEFINITION_TEMPLATE = Template(
        dedent(
            """
            Term: $term
            NCT IDs: $nct_ids

            Context passages:
            $context

            Generate one concise composition-centered definition for "$term"
            using only explicitly supported evidence. Preserve only active
            composition, molecular class, structural family, biochemical
            identity, origin, or another property useful for class mapping.

            Do not infer composition from modality, target, therapeutic use,
            indication, delivery vehicle, route, or product name. For a
            composite product, describe the complete active product rather than
            an ancillary carrier, linker, excipient, payload vehicle, or
            delivery system.

            Begin directly with the requested intervention name. Append the
            ClinicalTrials.gov citation and place the final period afterward.
            """
        ).strip()
    )

    DRUG_CLASS_RETRIEVAL_QUERY_SYSTEM_PROMPT = dedent(
        """
        You are a biomedical ontology curator generating semantic retrieval
        queries for a molecular, structural, biochemical, compositional, and
        origin-based drug-class hierarchy.

        Generate one primary query and up to two optional complementary
        queries. Queries are retrieval aids, not final mappings or ontology
        assertions. Classify what the complete active intervention is, not what
        it treats.

        The primary query must be the most specific class whose complete
        meaning is explicitly supported. Do not return indication, therapeutic
        function, target, route, delivery system, dosage form, product name, or
        development code. Use an empty string when unsupported.

        Require explicit evidence for DNA, protein, antibody, peptide, steroid,
        amino-acid derivative, nucleoside or nucleotide derivative, natural
        product, small molecule, or biochemical drug identity.

        Modality is not composition. Gene therapy, genome editing, cell
        therapy, viral-vector delivery, antibody targeting, or
        lipid-nanoparticle delivery does not by itself establish a class.

        Alternative queries must be complementary and positively supported,
        not lexical paraphrases, and must not describe only an ancillary
        component. Do not generate parents or siblings merely to fill the list.

        Return exactly one valid JSON object:
        {
          "primary_query": "best supported drug-class query or empty string",
          "alternative_queries": []
        }
        Return no Markdown, explanation, comments, or extra keys.
        """
    ).strip()

    DRUG_CLASS_RETRIEVAL_QUERY_TEMPLATE = Template(
        dedent(
            f"""
            Composition-centered intervention definition:
            $definition

            {DRUG_CLASS_HIERARCHY_AWARENESS}

            Generate drug-class retrieval queries. Classify the complete active
            product, not an ancillary component. Do not infer composition from
            modality, target, therapeutic use, indication, route, delivery
            system, or product name. Do not create broad or root classes merely
            to form an apparent hierarchy.

            Return exactly:
            {{
              "primary_query": "best supported drug-class query or empty string",
              "alternative_queries": []
            }}
            """
        ).strip()
    )

    DRUG_CLASS_MAPPING_SYSTEM_PROMPT = dedent(
        """
        You are a biomedical ontology curator mapping an intervention to a
        molecular, structural, biochemical, compositional, or origin-based
        drug class.

        Use the intervention definition as evidence. Retrieval queries are
        search aids, candidate definitions state class requirements, and rank
        or similarity are retrieval signals rather than biomedical evidence.

        Copy one exact supplied candidate name. Select the most specific class
        whose complete meaning is positively supported. Do not infer
        composition from indication, therapeutic function, target, route,
        delivery system, modality, product name, or development code. Absence
        of contradiction is not positive support.

        Classify the complete active product rather than an isolated carrier,
        linker, excipient, vector, delivery vehicle, or inactive component. Do
        not treat sibling classes as interchangeable.

        Select a child only when every additional restriction is supported. If
        a child is unsupported but its parent is supported, select the parent.
        Return NONE_OF_THE_ABOVE when no candidate is compatible.

        COMPATIBILITY AND MATCH-TYPE CONSISTENCY

        Choose compatibility and match_type as one inseparable pair.

        The only permitted pairs are:
        - directly_compatible + exact
        - directly_compatible + equivalent_label
        - broader_compatible + broader_available
        - related_compatible + nearest_available
        - none + none

        No other compatibility and match_type combination is valid.

        If selected_term is NONE_OF_THE_ABOVE, compatibility and match_type
        must both be none. If compatibility is none or match_type is none,
        selected_term must be NONE_OF_THE_ABOVE.

        Ground the reason in explicit compositional evidence and do not
        introduce a new composition, origin, modality, component, or class.

        Return exactly one valid JSON object with four string keys:
        {
          "selected_term": "exact candidate name or NONE_OF_THE_ABOVE",
          "compatibility": "directly_compatible, broader_compatible, related_compatible, or none",
          "match_type": "exact, equivalent_label, broader_available, nearest_available, or none",
          "reason": "one concise grounded sentence"
        }
        Return no Markdown, comments, explanations, or additional keys.
        """
    ).strip()

    DRUG_CLASS_MAPPING_TEMPLATE = Template(
        dedent(
            """
            Composition-centered intervention definition:
            $definition

            Drug-class retrieval queries:
            $retrieval_queries

            Drug-class ontology candidates:
            $candidates

            Retrieval queries are search aids and not independent evidence.
            Candidate definitions describe class requirements and must not be
            rewritten as intervention-specific facts.

            Evaluate each candidate's complete meaning, remove unsupported
            composition or origin, remove ancillary-component classes, select
            the most specific compatible candidate for the complete active
            product, assign compatibility and match_type, and write one concise
            grounded reason.

            If no candidate remains compatible, return NONE_OF_THE_ABOVE with
            compatibility and match_type both set to none.

            Before returning the result, verify that compatibility and
            match_type form exactly one of these permitted pairs:
            - directly_compatible + exact
            - directly_compatible + equivalent_label
            - broader_compatible + broader_available
            - related_compatible + nearest_available
            - none + none

            Do not return any other compatibility and match_type pairing.

            Return exactly one JSON object with four keys:
            {
              "selected_term": "exact candidate name or NONE_OF_THE_ABOVE",
              "compatibility": "directly_compatible, broader_compatible, related_compatible, or none",
              "match_type": "exact, equivalent_label, broader_available, nearest_available, or none",
              "reason": "one concise grounded sentence"
            }
            """
        ).strip()
    )

    # =========================================================================
    # NCIT SEARCH-TERM EXTRACTION
    # =========================================================================

    NCIT_SEARCH_TERM_EXTRACTION_SYSTEM_PROMPT = dedent(
        """
        You are a biomedical nomenclature expert. Produce the single NCIt
        search term most likely to retrieve the correct concept for the
        requested drug or therapy intervention.

        IDENTITY FIRST

        Return the canonical identity explicitly supported by the supplied
        evidence. Never replace the requested intervention with a comparator,
        co-intervention, regimen partner, payload, target, metabolite, carrier,
        delivery vehicle, study arm, sponsor, indication, or mechanism.

        STRIP PRESENTATION DETAILS

        Remove details that express presentation rather than identity:
        - strength and dose;
        - frequency and schedule;
        - route and infusion duration;
        - packaging and container information;
        - dosage-form words such as injection, tablet, capsule, solution,
          suspension, vial, and prefilled syringe;
        - study-arm, cohort, group, and regimen labels.

        PRESERVE IDENTITY-DEFINING DETAILS

        Preserve details when they distinguish the therapeutic identity:
        - salt or ester form;
        - prodrug identity;
        - stereochemical descriptor;
        - explicitly named combination components;
        - antibody-drug conjugate or fusion-protein identity;
        - radiolabel;
        - autologous or allogeneic cellular origin when identity-defining;
        - CAR-T identity;
        - gene-editing modality;
        - development code when no generic or canonical name is explicitly
          evidenced.

        FORMULATION QUALIFIERS

        Strip qualifiers such as qualifier-A, qualifier-B, qualifier-C,
        qualifier-D, qualifier-E, and qualifier-F only when the evidence
        explicitly names the underlying active substance and the qualifier is
        not required to distinguish the product. Otherwise preserve it.

        BRAND, CODE, AND ACTIVE-SUBSTANCE RULE

        Convert a brand name, formulation name, or development code to a
        standard or active-entity name only when the supplied evidence
        explicitly establishes that equivalence.

        If the trial label has the form "standard-name (brand-name)", prefer the
        standard name. Treat arm or cohort labels as presentation details, not
        aliases.

        For a complete multi-product regimen, do not collapse the regimen to a
        single component. Return the complete identity only when this stage is
        explicitly asked for one term. Component-wise NCIt resolution must be
        performed separately by the calling pipeline.

        EVIDENCE CONSTRAINT

        Use only evidence explicitly attributed to the requested intervention.
        Do not infer identity from indication, target, mechanism, sponsor,
        publication background, comparator, or another trial intervention.

        If no simpler canonical identity is supported, return the author term
        after stripping only unambiguous presentation details.

        OUTPUT

        Return one bare search term only. Return no quotes, JSON, Markdown,
        explanation, NCIt code, category, or trailing punctuation.
        """
    ).strip()

    NCIT_SEARCH_TERM_EXTRACTION_TEMPLATE = Template(
        dedent(
            """
            Intervention term: $term
            NCT IDs: $nct_ids

            Identity evidence:
            $context

            Return the single evidence-supported NCIt search term for this
            intervention. Preserve identity-defining qualifiers and remove only
            unambiguous presentation details.
            """
        ).strip()
    )

    # =========================================================================
    # NCIt candidate list extraction
    # =========================================================================

    NCIT_CANDIDATE_LIST_SYSTEM_PROMPT = dedent(
        """
        Biomedical nomenclature resolver. Given an intervention term and
        clinical trial evidence, return NCIt search names as JSON.

        STEP 1 — classify from the AUTHOR TERM alone:
          single_product  one intervention/entity/therapy (even if co-administered)
          regimen         two+ entities joined by +/plus///combined with, or a
                          recognised multi-component acronym (ACRONYM-A, ACRONYM-B …)
          composite_product  one product with multiple declared active entities
          class_request   the author term names a THERAPEUTIC CLASS or a
                          FUNCTIONAL DESCRIPTION rather than a specific entity
                          (e.g. “TargetX directed therapy”, “anti-TargetY agent”,
                          “PathwayZ inhibitor”, “class-based inhibitor”).
                          When class_request: emit the single concrete entity
                          listed in the trial record as requested_identity so
                          NCIt can be searched on it. Do not split or enumerate
                          multiple class members.

        STEP 2 — read the ENTIRE term including all parenthetical groups before
        making any splitting decision. Then emit one search_term per distinct
        searchable therapeutic entity.

          KEEP  standard name · proprietary code (only when no standard name exists) · brand name ·
                salt/ester that distinguishes the entity · carrier qualifier
                that changes the approved identity (e.g. carrier-prefix, formulation-qualifier) ·
                stereochemical prefix · radionuclide/isotope · targeting molecule in a conjugate ·
                biological origin (autologous, allogeneic) · CAR/gene-edit target ·
                FULL CAR-T construct descriptor including all slash-separated
                components (target antigen, costimulatory domain, transgene)
                that precede the cell-type suffix — e.g. “ConstructA/DomainB/TransgeneC”
                in “ConstructA/DomainB/TransgeneC CAR T-lymphocytes” is a single
                identity-defining construct code; preserve it intact

          DROP  route (IV SC PO oral …) · dose/strength · dosage form
                (injection tablet solution …) · schedule and timing · trial
                and arm labels · development code when the standard name is present ·
                linker or chelator that is not itself an active entity ·
                parenthetical groups that are synonyms of the whole term rather
                than names of additional entities · trailing plural -s when
                the singular form is the standard database entry (e.g.
                "EntityA cells" → "EntityA cell", "EntityB molecules" → "EntityB molecule")

          Splitting rules:
          - Connectors +/plus/combined with/with between named entities →
            split into one entry per entity.
          - A slash between complete entity names → split; a slash inside a
            single entity name (e.g. qualifier-EntityA) → do not split.
          - A slash WITHIN a CAR-T construct descriptor (separating antigen /
            domain / transgene components before the cell-type word) → do NOT
            split; the whole descriptor is the product identity.
            Signal: term ends in a recognised cell-type suffix (e.g. "CAR T",
            "CAR T-cell", "CAR T-lymphocyte", or similar).
          - A slash separating active entities that are co-formulated into
            one physical product (single vial/syringe, one brand name) →
            composite_product, not regimen. When a brand name appears in a
            parenthetical, that confirms composite_product. Use the full
            slash-joined name (with parentheticals) as matched_trial_name and
            emit each active entity as requested_composite_component.
          - Recognised multi-entity acronym → resolve to its named components.
          - Conjugate or labelled product (label-linker-targeting molecule) →
            emit each named active component (label entity, linker
            when it is an active entity, targeting molecule) as a separate
            entry.
          - Parentheticals listing synonyms of the whole term → do not split
            as if they were additional entities; use them only to confirm identity.
          - Comma/and-separated list of distinct entities in an
            alternative name → split each entity into its own entry.
          - Prose modifiers (dose, schedule, route, "modified", arm labels) →
            strip entirely; do not treat as component boundaries.

        OUTPUT — valid JSON only, no markdown:
        {"resolution_kind":"...","matches":[{
          "search_term":"<INN, code, or brand>",
          "matched_trial_name":"<verbatim from Intervention N exact name>",
          "relationship":"normalized_name|exact_name|combination_component|alias",
          "role":"requested_identity|requested_regimen_component|requested_composite_component",
          "evidence":"<one sentence>"
        }]}
        matched_trial_name must be copied verbatim — never invented.
        single_product: role=requested_identity, no combination partners.
        regimen: one entry per entity, role=requested_regimen_component.
        Order: stripped canonical first, then exact NCT name if different, then aliases.

        EXAMPLES
        [1] term="XYZ" | regimen acronym — entities: EntityA, EntityB, EntityC
        {"resolution_kind":"regimen","matches":[
          {"search_term":"EntityA","matched_trial_name":"XYZ","relationship":"combination_component","role":"requested_regimen_component","evidence":"First named entity of the XYZ regimen."},
          {"search_term":"EntityB","matched_trial_name":"XYZ","relationship":"combination_component","role":"requested_regimen_component","evidence":"Second named entity of the XYZ regimen."},
          {"search_term":"EntityC","matched_trial_name":"XYZ","relationship":"combination_component","role":"requested_regimen_component","evidence":"Third named entity of the XYZ regimen."}]}

        [2] term="CODE-A + modified EntityB/carrier-EntityC (Treatment Group B)"
        Three entities joined by + and /. "modified" and "Treatment Group B" are trial labels — drop.
        {"resolution_kind":"regimen","matches":[
          {"search_term":"CODE-A","matched_trial_name":"CODE-A + modified EntityB/carrier-EntityC (Treatment Group B)","relationship":"combination_component","role":"requested_regimen_component","evidence":"Proprietary code; no INN. Arm label stripped."},
          {"search_term":"EntityB","matched_trial_name":"CODE-A + modified EntityB/carrier-EntityC (Treatment Group B)","relationship":"combination_component","role":"requested_regimen_component","evidence":"INN; modifier stripped."},
          {"search_term":"carrier-EntityC","matched_trial_name":"CODE-A + modified EntityB/carrier-EntityC (Treatment Group B)","relationship":"combination_component","role":"requested_regimen_component","evidence":"Carrier prefix kept: distinct NCIt entry from plain EntityC."}]}

        [3] term="CODE-X" | evidence "CODE-X for injection"
        {"resolution_kind":"single_product","matches":[
          {"search_term":"CODE-X","matched_trial_name":"CODE-X for injection","relationship":"normalized_name","role":"requested_identity","evidence":"Dosage form stripped."},
          {"search_term":"CODE-X for injection","matched_trial_name":"CODE-X for injection","relationship":"exact_name","role":"requested_identity","evidence":"Exact NCT name as fallback."}]}

        [4] term="EntityA (SC) (CODE-Y)" — (SC)=route drop; (CODE-Y)=code redundant given INN drop
        {"resolution_kind":"single_product","matches":[
          {"search_term":"EntityA","matched_trial_name":"EntityA (SC) (CODE-Y)","relationship":"normalized_name","role":"requested_identity","evidence":"Route and redundant code dropped; INN is the identity."}]}

        [5] term="alternate day EntityA-salt with an equivalent elemental EntityA dose of 65 mg per pill (alternate day)"
        Two active entities named; dose, schedule, and route context stripped.
        {"resolution_kind":"regimen","matches":[
          {"search_term":"EntityA-salt","matched_trial_name":"alternate day EntityA-salt with an equivalent elemental EntityA dose of 65 mg per pill (alternate day)","relationship":"combination_component","role":"requested_regimen_component","evidence":"Salt form retained as identity; dose and schedule stripped."},
          {"search_term":"Elemental EntityA","matched_trial_name":"alternate day EntityA-salt with an equivalent elemental EntityA dose of 65 mg per pill (alternate day)","relationship":"combination_component","role":"requested_regimen_component","evidence":"Active entity explicitly named; dose stripped."}]}

        [6] term="LabelX-Chelator-TargetingMolecule (LX-Chelator-TargetingMolecule; [LX]-Chelator-TargetingMolecule)"
        Parentheticals are synonyms of the whole conjugate — not extra entities. Three active components.
        {"resolution_kind":"composite_product","matches":[
          {"search_term":"LabelX","matched_trial_name":"LabelX-Chelator-TargetingMolecule (LX-Chelator-TargetingMolecule; [LX]-Chelator-TargetingMolecule)","relationship":"combination_component","role":"requested_composite_component","evidence":"Label/radionuclide component."},
          {"search_term":"Chelator","matched_trial_name":"LabelX-Chelator-TargetingMolecule (LX-Chelator-TargetingMolecule; [LX]-Chelator-TargetingMolecule)","relationship":"combination_component","role":"requested_composite_component","evidence":"Chelating agent component."},
          {"search_term":"TargetingMolecule","matched_trial_name":"LabelX-Chelator-TargetingMolecule (LX-Chelator-TargetingMolecule; [LX]-Chelator-TargetingMolecule)","relationship":"combination_component","role":"requested_composite_component","evidence":"Targeting molecule component."}]}

        [7] term="Enhancer-variant/EntityA/EntityB (EntityA and EntityB enhancer; EntityA, EntityB, and Enhancer-variant; EntityA/EntityB/Enhancer-variant; BrandName)"
        The brand name BrandName confirms one co-formulated product (single vial) — not a regimen.
        Parentheticals are synonyms of the whole product — not extra entities.
        {"resolution_kind":"composite_product","matches":[
          {"search_term":"EntityA","matched_trial_name":"Enhancer-variant/EntityA/EntityB (EntityA and EntityB enhancer; EntityA, EntityB, and Enhancer-variant; EntityA/EntityB/Enhancer-variant; BrandName)","relationship":"combination_component","role":"requested_composite_component","evidence":"Active component co-formulated in BrandName."},
          {"search_term":"EntityB","matched_trial_name":"Enhancer-variant/EntityA/EntityB (EntityA and EntityB enhancer; EntityA, EntityB, and Enhancer-variant; EntityA/EntityB/Enhancer-variant; BrandName)","relationship":"combination_component","role":"requested_composite_component","evidence":"Active component co-formulated in BrandName."},
          {"search_term":"Enhancer-variant","matched_trial_name":"Enhancer-variant/EntityA/EntityB (EntityA and EntityB enhancer; EntityA, EntityB, and Enhancer-variant; EntityA/EntityB/Enhancer-variant; BrandName)","relationship":"combination_component","role":"requested_composite_component","evidence":"Permeation enhancer component co-formulated in BrandName."}]}

        [8] term="TargetX directed therapy" | evidence: trial entity is "CODE-Z" (a TargetX-targeting agent)
        The author term names a therapeutic class, not a specific entity.
        {"resolution_kind":"class_request","matches":[
          {"search_term":"CODE-Z","matched_trial_name":"CODE-Z","relationship":"class_member","role":"requested_identity","evidence":"Concrete TargetX-directed entity listed in the trial; used as NCIt anchor for the class."}]}

        [9] term="ConstructA/DomainB/TransgeneC CAR T-lymphocytes" | evidence: intervention exact name is "ConstructA/DomainB/TransgeneC CAR T-lymphocytes"
        The slash-separated prefix "ConstructA/DomainB/TransgeneC" is a CAR construct descriptor
        (safety switch / costimulatory domain / transgene) — a single identity-defining code, not separate
        entities. The cell-type suffix "CAR T-lymphocytes" confirms this is one engineered cell product.
        Do NOT split on slashes; do NOT strip the construct prefix; keep the whole name intact.
        {"resolution_kind":"single_product","matches":[
          {"search_term":"ConstructA/DomainB/TransgeneC CAR T-lymphocyte","matched_trial_name":"ConstructA/DomainB/TransgeneC CAR T-lymphocytes","relationship":"normalized_name","role":"requested_identity","evidence":"Trailing plural stripped; full construct descriptor preserved intact."},
          {"search_term":"ConstructA/DomainB/TransgeneC CAR T-lymphocytes","matched_trial_name":"ConstructA/DomainB/TransgeneC CAR T-lymphocytes","relationship":"exact_name","role":"requested_identity","evidence":"Exact NCT intervention name as fallback."}]}
        """
    ).strip()

    NCIT_CANDIDATE_LIST_TEMPLATE = Template(
        dedent(
            """
            Intervention term: $term
            NCT IDs: $nct_ids

            Evidence (intervention names and context from ClinicalTrials.gov):
            $context

            Step 1 — inspect the author-provided term above.
            Decide resolution_kind from the TERM alone, not from the evidence:
              - Does the term explicitly name multiple entities? → regimen
              - Does the term name one product (even if given with others)? → single_product

            Step 2 — use the evidence to find matched_trial_name.
              - Copy matched_trial_name verbatim from an "Intervention N exact
                name" line in the evidence.
              - For single_product, return only the requested product and its
                aliases. Do not return combination partners.
              - For regimen, return one entry per component explicitly requested
                in the author term.

            Step 3 — expand list-valued alternative names.
              - If any alternative name in the evidence is a comma- or
                "and"-separated list of distinct biological entities (antigens,
                gene names, protein names, cell types, etc.), emit each entity
                as its own separate match entry with its own search_term.
              - Never put multiple entities into a single search_term value.
              - All entries expanded from the same intervention share the same
                matched_trial_name (the exact intervention name).

            Return the structured JSON object. Role is required on every match.
            """
        ).strip()
    )
