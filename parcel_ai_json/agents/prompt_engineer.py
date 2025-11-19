"""Dynamic Prompt Engineering Agent for context-aware Grounded-SAM detection."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PropertyContext:
    """Context information about a property for prompt generation."""

    center_lat: float
    center_lon: float
    lot_size_sqft: Optional[float] = None
    property_type: Optional[str] = None  # residential, commercial, industrial
    address: Optional[str] = None
    climate_zone: Optional[str] = None  # warm, temperate, cold
    region: Optional[str] = None  # southwest, northeast, etc.


@dataclass
class PromptResult:
    """Result from prompt generation with metadata."""

    prompts: List[str]
    reasoning: str
    confidence_score: float
    excluded_prompts: List[str]
    climate_context: str
    property_analysis: str


class PromptEngineeringAgent:
    """Claude-powered agent for generating context-aware detection prompts."""

    # Default prompts as fallback
    DEFAULT_PROMPTS = [
        "driveway",
        "patio",
        "deck",
        "shed",
        "gazebo",
        "pergola",
        "hot tub",
        "fire pit",
        "pool house",
        "dog house",
        "playground equipment",
        "trampoline",
        "basketball hoop",
        "above ground pool",
        "boat",
        "RV",
        "trailer",
        "carport",
        "greenhouse",
        "chicken coop",
    ]

    def __init__(self, api_key: Optional[str] = None, use_agent: bool = True):
        """Initialize prompt engineering agent.

        Args:
            api_key: Anthropic API key (if None, falls back to env var)
            use_agent: If False, returns default prompts (for testing/fallback)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.use_agent = use_agent and self.api_key is not None

        if self.use_agent:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("PromptEngineeringAgent initialized with Claude API")
            except ImportError:
                logger.warning(
                    "anthropic package not installed - using default prompts"
                )
                self.use_agent = False
        else:
            logger.info("PromptEngineeringAgent using default prompts (no API key)")

    def generate_prompts(
        self, context: PropertyContext, max_prompts: int = 15
    ) -> PromptResult:
        """Generate context-aware detection prompts.

        Args:
            context: Property context information
            max_prompts: Maximum number of prompts to generate

        Returns:
            PromptResult with optimized prompts and metadata
        """
        if not self.use_agent:
            logger.info("Using default prompts (agent disabled)")
            return PromptResult(
                prompts=self.DEFAULT_PROMPTS[:max_prompts],
                reasoning="Agent disabled - using default prompts",
                confidence_score=0.5,
                excluded_prompts=[],
                climate_context="Unknown",
                property_analysis="No agent analysis",
            )

        try:
            # Build context description
            context_desc = self._build_context_description(context)

            # Call Claude API
            logger.info(f"Calling Claude API for prompt generation: {context_desc}")

            import anthropic

            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are an expert in property analysis from satellite imagery.

Your task: Generate the most relevant {max_prompts} detection prompts for Grounded-SAM
(a text-prompted object detection model) based on property context.

Property Context:
{context_desc}

Guidelines:
1. Prioritize features likely to exist based on:
   - Climate/geography (pools in warm climates, garages in cold)
   - Property type (residential amenities vs commercial features)
   - Lot size (luxury estates have more features)
   - Region (RVs common in southwest, enclosed porches in northeast)

2. Be specific and concrete:
   - Use "inground swimming pool" not just "pool"
   - Use "outdoor kitchen" not "kitchen"
   - Use "multi-car garage" not "garage"

3. Exclude irrelevant features:
   - No snow equipment in Arizona
   - No pools in Alaska
   - No agricultural features in urban lots

4. Prioritize by likelihood (most likely first)

Return your response in this exact JSON format:
{{
  "prompts": ["prompt1", "prompt2", ...],
  "reasoning": "Brief explanation of why these prompts were chosen",
  "confidence_score": 0.0-1.0,
  "excluded_prompts": ["excluded1", "excluded2"],
  "climate_context": "Brief climate analysis",
  "property_analysis": "Brief property type analysis"
}}

IMPORTANT: Return ONLY valid JSON, no markdown formatting.""",
                    }
                ],
            )

            # Parse response
            import json

            response_text = message.content[0].text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                # Extract JSON from markdown code block
                lines = response_text.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                response_text = "\n".join(json_lines)

            result_data = json.loads(response_text)

            result = PromptResult(
                prompts=result_data["prompts"][:max_prompts],
                reasoning=result_data["reasoning"],
                confidence_score=result_data["confidence_score"],
                excluded_prompts=result_data.get("excluded_prompts", []),
                climate_context=result_data.get("climate_context", "N/A"),
                property_analysis=result_data.get("property_analysis", "N/A"),
            )

            logger.info(
                f"Generated {len(result.prompts)} prompts "
                f"(confidence: {result.confidence_score:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating prompts with Claude API: {e}")
            logger.info("Falling back to default prompts")
            return PromptResult(
                prompts=self.DEFAULT_PROMPTS[:max_prompts],
                reasoning=f"Fallback due to error: {str(e)}",
                confidence_score=0.5,
                excluded_prompts=[],
                climate_context="Error occurred",
                property_analysis="Error occurred",
            )

    def _build_context_description(self, context: PropertyContext) -> str:
        """Build human-readable context description for Claude."""
        parts = []

        # Location
        parts.append(f"Location: {context.center_lat:.4f}, {context.center_lon:.4f}")

        # Infer climate from latitude
        climate = self._infer_climate(context.center_lat)
        parts.append(f"Climate: {climate}")

        # Infer region from coordinates
        region = self._infer_region(context.center_lat, context.center_lon)
        if region:
            parts.append(f"Region: {region}")

        # Lot size
        if context.lot_size_sqft:
            acres = context.lot_size_sqft / 43560
            size_desc = "small" if acres < 0.25 else "large" if acres > 1 else "medium"
            parts.append(
                f"Lot size: {context.lot_size_sqft:,.0f} sqft "
                f"({acres:.2f} acres, {size_desc})"
            )

        # Property type
        if context.property_type:
            parts.append(f"Property type: {context.property_type}")

        # Address
        if context.address:
            parts.append(f"Address: {context.address}")

        return "\n".join(parts)

    def _infer_climate(self, latitude: float) -> str:
        """Infer climate zone from latitude."""
        abs_lat = abs(latitude)
        if abs_lat < 23.5:
            return "tropical"
        elif abs_lat < 35:
            return "warm/subtropical"
        elif abs_lat < 50:
            return "temperate"
        else:
            return "cold"

    def _infer_region(self, latitude: float, longitude: float) -> str:
        """Infer US region from coordinates (simplified)."""
        # US-specific heuristic
        if 24 < latitude < 50 and -125 < longitude < -65:
            if longitude < -100:
                if latitude > 40:
                    return "northwest"
                else:
                    return "southwest"
            else:
                if latitude > 40:
                    return "northeast"
                else:
                    return "southeast"
        return "unknown"
