import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── NCIt identity-resolution schema ──────────────────────────────────────────

ResolutionKind = Literal[
    "single_product",
    "regimen",
    "composite_product",
    "single_procedure",
    "class_request",
    "unresolved",
]

Relationship = Literal[
    "exact_name",
    "normalized_name",
    "generic_name",
    "brand_name",
    "development_code",
    "explicit_alias",
    "active_substance",
    "combination_component",
    "procedure_in_intervention",
    "class_member",
    "alias",
]

IdentityRole = Literal[
    "requested_identity",
    "requested_regimen_component",
    "requested_composite_component",
    "co_intervention",
    "excluded_context",
]

ComponentType = Literal[
    "drug",
    "biological",
    "cell_therapy",
    "gene_therapy",
    "radiopharmaceutical",
    "procedure",
    "device",
    "other",
]


class NCItIdentityMatch(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    search_term: str = Field(min_length=1)
    matched_trial_name: str = Field(min_length=1)
    relationship: Relationship
    role: IdentityRole
    component_type: ComponentType = "drug"
    evidence: str = ""


class NCItIdentityResolution(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resolution_kind: ResolutionKind
    matches: list[NCItIdentityMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution(self) -> "NCItIdentityResolution":
        if self.resolution_kind == "unresolved" and self.matches:
            raise ValueError(
                "Unresolved identity cannot contain matches."
            )
        if self.resolution_kind != "unresolved" and not self.matches:
            raise ValueError(
                "A resolved identity must contain at least one match."
            )

        allowed_roles: dict[str, set[str]] = {
            "single_product": {"requested_identity"},
            "single_procedure": {"requested_identity"},
            "regimen": {"requested_regimen_component"},
            "composite_product": {"requested_composite_component"},
            "class_request": {"requested_identity"},
        }
        passive_roles = {"co_intervention", "excluded_context"}
        expected = allowed_roles.get(self.resolution_kind, set())

        for match in self.matches:
            if match.role in passive_roles:
                continue
            if expected and match.role not in expected:
                raise ValueError(
                    f"Role {match.role!r} is incompatible with "
                    f"resolution_kind {self.resolution_kind!r}."
                )
        return self


def _extract_reference_title(citation: str) -> str:
    """
    Extract the article title from an NLM-style citation string.
    Format: "Author A, Author B. Title of article. Journal. Year;vol:pages."
    Returns text between the first and second ". " delimiter.
    Falls back to the full citation if the pattern does not match.
    """
    citation = citation.strip()
    if not citation:
        return ""
    first_dot = citation.find(". ")
    if first_dot == -1:
        return citation
    after_authors = citation[first_dot + 2:]
    second_dot = after_authors.find(". ")
    if second_dot == -1:
        return after_authors.strip()
    return after_authors[:second_dot].strip()


class InternalDrugClassMapping(BaseModel):
    """Internal result returned by the drug-class mapping call."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    selected_term: str = Field(
        description=(
            "Exact supplied drug-class candidate name, or "
            "NONE_OF_THE_ABOVE when no compatible candidate exists."
        )
    )

    compatibility: Literal[
        "directly_compatible",
        "broader_compatible",
        "related_compatible",
        "none",
    ]

    match_type: Literal[
        "exact",
        "equivalent_label",
        "broader_available",
        "nearest_available",
        "none",
    ]

    reason: str

    @field_validator("selected_term", "reason")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_internal_consistency(
        self,
    ) -> "InternalDrugClassMapping":
        valid_combinations = {
            "directly_compatible": {
                "exact",
                "equivalent_label",
            },
            "broader_compatible": {
                "broader_available",
            },
            "related_compatible": {
                "nearest_available",
            },
            "none": {
                "none",
            },
        }

        if self.match_type not in valid_combinations[
            self.compatibility
        ]:
            raise ValueError(
                "match_type is inconsistent with compatibility"
            )

        selected_is_none = (
            self.selected_term == "NONE_OF_THE_ABOVE"
        )

        if self.compatibility == "none":
            if not selected_is_none:
                raise ValueError(
                    "compatibility=none requires "
                    "selected_term=NONE_OF_THE_ABOVE"
                )
        elif selected_is_none:
            raise ValueError(
                "A compatible mapping must select a supplied "
                "drug-class candidate"
            )

        return self

class InternalOntologyMapping(BaseModel):
    """Internal result returned by the third LLM call."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    selected_term: str = Field(
        description=(
            "Exact supplied ontology candidate name, or "
            "NONE_OF_THE_ABOVE when no compatible candidate exists."
        )
    )

    compatibility: Literal[
        "directly_compatible",
        "broader_compatible",
        "related_compatible",
        "none",
    ]

    match_type: Literal[
        "exact",
        "equivalent_label",
        "broader_available",
        "cross_axis_therapeutic_match",
        "nearest_available",
        "none",
    ]

    reason: str

    @field_validator("selected_term", "reason")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_internal_consistency(
        self,
    ) -> "InternalOntologyMapping":
        valid_combinations = {
            "directly_compatible": {
                "exact",
                "equivalent_label",
            },
            "broader_compatible": {
                "broader_available",
            },
            "related_compatible": {
                "nearest_available",
                "cross_axis_therapeutic_match",
            },
            "none": {
                "none",
            },
        }

        if self.match_type not in valid_combinations[self.compatibility]:
            raise ValueError(
                "match_type is inconsistent with compatibility"
            )

        selected_is_none = (
            self.selected_term == "NONE_OF_THE_ABOVE"
        )

        if self.compatibility == "none":
            if not selected_is_none:
                raise ValueError(
                    "compatibility=none requires "
                    "selected_term=NONE_OF_THE_ABOVE"
                )
        elif selected_is_none:
            raise ValueError(
                "A compatible mapping must select a supplied "
                "ontology candidate"
            )

        return self


# ── NCT Drug Context Models ───────────────────────────────────────────────────

class DrugClassRetrievalSuggestion(BaseModel):
    """
    Molecular, structural, biochemical, or compositional retrieval
    queries extracted from a composition-centered intervention
    definition.

    These queries improve candidate retrieval and do not assert
    ontology parent-child relationships.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    primary_query: str = Field(
        description=(
            "The single best and most specific molecular, structural, "
            "biochemical, or compositional drug-class retrieval query "
            "explicitly supported by the intervention definition. "
            "Do not infer composition from modality, target, indication, "
            "delivery system, route, product name, or development code. "
            "Use an empty string when no drug-class query is supported."
        )
    )

    alternative_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Up to two complementary drug-class retrieval queries "
            "explicitly supported by the intervention definition. "
            "They are retrieval aids and do not assert ontology "
            "parent-child relationships."
        )
    )

    @field_validator("primary_query")
    @classmethod
    def normalize_primary_query(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("alternative_queries", mode="before")
    @classmethod
    def validate_alternatives_input(
        cls,
        value: object,
    ) -> object:
        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                "alternative_queries must be a JSON array"
            )

        return value

    @field_validator("alternative_queries")
    @classmethod
    def normalize_alternative_queries(
        cls,
        values: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(str(value).split()).strip()

            if not normalized:
                continue

            key = normalized.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result[:2]

    @model_validator(mode="after")
    def remove_primary_duplicate(
        self,
    ) -> "DrugClassRetrievalSuggestion":
        if not self.primary_query:
            self.alternative_queries = []
            return self

        primary_key = self.primary_query.casefold()

        self.alternative_queries = [
            query
            for query in self.alternative_queries
            if query.casefold() != primary_key
        ]

        return self

class TherapeuticCategorySuggestion(BaseModel):
    """
    Functional therapeutic or pharmacological categories extracted
    from an evidence-grounded intervention definition.

    Every field must be present, but its value may be empty when the
    definition does not independently support that category level.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    specific_category: str = Field(
        description=(
            "The primary and most specific supported reusable functional "
            "therapeutic or pharmacological category. This is the preferred "
            "retrieval query. It must not be a product modality, platform, "
            "cell type, molecular target, indication, product name, route, "
            "dose, dosage form, or development status. Use an empty string "
            "when no functional category is supported."
        ),
    )

    broad_category: str = Field(
        description=(
            "An optional category that is meaningfully broader than "
            "specific_category and belongs to the same functional hierarchy "
            "path. Use an empty string when specific_category is empty, when "
            "no valid broader category is supported, or when the value would "
            "duplicate specific_category."
        ),
    )

    root_category: str = Field(
        description=(
            "An optional broadest informative ancestor of another populated "
            "category field. It must belong to the same functional hierarchy "
            "path. Use an empty string when no informative ancestor is "
            "supported or when the value would be redundant."
        ),
    )

    @field_validator(
        "specific_category",
        "broad_category",
        "root_category",
    )
    @classmethod
    def normalize_whitespace(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def normalize_category_levels(
        self,
    ) -> "TherapeuticCategorySuggestion":
        # Broad and root cannot exist without a specific category.
        if not self.specific_category:
            self.broad_category = ""
            self.root_category = ""
            return self

        seen: set[str] = set()

        for field_name in (
            "specific_category",
            "broad_category",
            "root_category",
        ):
            value = getattr(self, field_name)

            if not value:
                continue

            key = value.casefold()

            if key in seen:
                setattr(self, field_name, "")
            else:
                seen.add(key)

        return self

class NCTIntervention(BaseModel):
    """A single drug or biological intervention from a trial."""

    name: str
    type: str | None = None
    description: str | None = None
    other_names: list[str] = Field(default_factory=list)


class NCTReference(BaseModel):
    """A single publication reference from a trial's referencesModule."""

    pmid: str = ""
    type: str = ""   # RESULT | BACKGROUND | DERIVED
    title: str = ""  # extracted from citation string


class NCTDrugContext(BaseModel):
    """
    Subset of a ClinicalTrials.gov record containing fields relevant
    to generating an intervention-centered definition.
    """

    nct_id: str

    official_title: str | None = None
    brief_title: str | None = None
    brief_summary: str | None = None
    detailed_description: str | None = None

    interventions: list[NCTIntervention] = Field(
        default_factory=list
    )

    conditions: list[str] = Field(
        default_factory=list
    )

    mesh_terms: list[str] = Field(
        default_factory=list
    )

    intervention_mesh_terms: list[str] = Field(
        default_factory=list
    )

    phases: list[str] = Field(
        default_factory=list
    )

    references: list[NCTReference] = Field(
        default_factory=list
    )

    @classmethod
    def from_api(cls, data: dict) -> "NCTDrugContext":
        """Extract intervention-relevant fields from an NCT API response."""

        protocol = data.get("protocolSection", {})
        derived = data.get("derivedSection", {})

        id_module = protocol.get(
            "identificationModule",
            {},
        )
        desc_module = protocol.get(
            "descriptionModule",
            {},
        )
        conditions_module = protocol.get(
            "conditionsModule",
            {},
        )
        design_module = protocol.get(
            "designModule",
            {},
        )
        arms_module = protocol.get(
            "armsInterventionsModule",
            {},
        )
        condition_browse = derived.get(
            "conditionBrowseModule",
            {},
        )
        intervention_browse = derived.get(
            "interventionBrowseModule",
            {},
        )

        return cls(
            nct_id=id_module.get("nctId", ""),
            official_title=id_module.get("officialTitle"),
            brief_title=id_module.get("briefTitle"),
            brief_summary=desc_module.get("briefSummary"),
            detailed_description=desc_module.get(
                "detailedDescription"
            ),
            interventions=[
                NCTIntervention(
                    name=item.get("name", ""),
                    type=item.get("type"),
                    description=item.get("description"),
                    other_names=item.get("otherNames", []),
                )
                for item in arms_module.get(
                    "interventions",
                    [],
                )
            ],
            conditions=conditions_module.get(
                "conditions",
                [],
            ),
            mesh_terms=[
                item.get("term", "")
                for item in condition_browse.get(
                    "meshes",
                    [],
                )
                if item.get("term")
            ],
            intervention_mesh_terms=[
                item.get("term", "")
                for item in intervention_browse.get(
                    "meshes",
                    [],
                )
                if item.get("term")
            ],
            phases=design_module.get(
                "phases",
                [],
            ),
            references=[
                NCTReference(
                    pmid=str(ref.get("pmid") or "").strip(),
                    type=str(ref.get("type") or "").strip(),
                    title=_extract_reference_title(
                        str(ref.get("citation") or "")
                    ),
                )
                for ref in protocol.get(
                    "referencesModule", {}
                ).get("references", [])
                if ref.get("citation") or ref.get("pmid")
            ],
        )


# ── API Request Models ───────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    query: str | list[str]


class LLMRequest(BaseModel):
    query: str
    response_format: type[BaseModel] | None = None


# ── Config Models ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to `default` when unset or blank."""
    raw = os.getenv(name, "")
    return int(raw) if raw.strip() else default


class LLMParameters:
    def __init__(self,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str,
        llm_app_id: str = "",
    ):
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_app_id = llm_app_id

    @classmethod
    def from_env(cls) -> "LLMParameters":
        return cls(
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_app_id=os.getenv("LLM_APP_ID", ""),
        )


class SapBERTParameters:
    def __init__(self,
        sapbert_model: str,
        sapbert_base_url: str,
        sapbert_api_key: str,
        sapbert_port: int = 7997,
    ):
        self.sapbert_model = sapbert_model
        self.sapbert_base_url = sapbert_base_url
        self.sapbert_api_key = sapbert_api_key
        self.sapbert_port = sapbert_port

    @classmethod
    def from_env(cls) -> "SapBERTParameters":
        return cls(
            sapbert_model=os.getenv("SAPBERT_MODEL", ""),
            sapbert_base_url=os.getenv("SAPBERT_BASE_URL", ""),
            sapbert_api_key=os.getenv("SAPBERT_API_KEY", ""),
            sapbert_port=_env_int("SAPBERT_PORT", 7997),
        )

class QdrantParameters:
    def __init__(
        self,
        url: str,
        collection_name: str,
        drug_class_collection_name: str,
        api_key: str = "",
    ):
        self.url = url
        self.collection_name = collection_name
        self.drug_class_collection_name = drug_class_collection_name
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "QdrantParameters":
        return cls(
            url=os.getenv("QDRANT_URL", ""),
            collection_name=os.getenv(
                "QDRANT_COLLECTION",
                "",
            ),
            drug_class_collection_name=os.getenv(
                "DRUG_CLASS_QDRANT_COLLECTION", ""
            ),
            api_key=os.getenv("QDRANT_API_KEY", ""),
        )


class PathParameters:
    def __init__(self,
        models_dir: Path = Path(__file__).parent.parent.parent / "models",
    ):
        self.models_dir = models_dir
        self.sapbert_model_path = models_dir / "sapbert"

    @classmethod
    def from_env(cls) -> "PathParameters":
        return cls(
            models_dir=Path(os.getenv("MODELS_DIR", "./models")),
        )


class ServerParameters:
    def __init__(self,
        server_port: int = 8000,
    ):
        self.server_port = server_port

    @classmethod
    def from_env(cls) -> "ServerParameters":
        return cls(
            server_port=_env_int("SERVER_PORT", 8000),
        )


class AppConfig:
    def __init__(self,
        llm: LLMParameters,
        sapbert: SapBERTParameters,
        paths: PathParameters,
        server: ServerParameters,
        qdrant: QdrantParameters,
    ):
        self.llm = llm
        self.sapbert = sapbert
        self.paths = paths
        self.server = server
        self.qdrant = qdrant

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            llm=LLMParameters.from_env(),
            sapbert=SapBERTParameters.from_env(),
            paths=PathParameters.from_env(),
            server=ServerParameters.from_env(),
            qdrant=QdrantParameters.from_env(),
        )
