from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
import base64
import cv2
from typing import List, Dict, Union
import time

from utils.CameraLocationMapping import CameraLocationMapping


class SceneAnalyzer:
    def __init__(self, api_key: str, camera_id: str = "Cam5"):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            max_tokens=500,
            temperature=0.2
        )
        self.memory = ConversationBufferMemory(memory_key="threat_analysis")
        self.threat_confirmation_count = 0
        self.analysis_complete = False
        self.last_analysis_time = None
        self.frames_analyzed = 0

        # Initialize camera location information
        self.camera_mapping = CameraLocationMapping()
        self.camera_id = camera_id
        self.location_info = self.camera_mapping.get_location_info(camera_id)

    def encode_image(self, frame) -> str:
        """Convert CV2 frame to base64"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_threat(self, frame, frame_number: int) -> Dict:
        """Analyze potential threat in the scene"""
        if self.analysis_complete:
            return {"status": "Analysis already completed"}

        current_time = time.time()
        base64_image = self.encode_image(frame)

        if self.frames_analyzed == 0:
            analysis = self._initial_threat_assessment(base64_image)
            self.frames_analyzed += 1
            self.last_analysis_time = current_time
            return {"status": "initial_assessment", "analysis": analysis}

        elif self.frames_analyzed < 3 and current_time - self.last_analysis_time >= 2:
            analysis = self._confirmatory_analysis(base64_image)
            self.frames_analyzed += 1
            self.last_analysis_time = current_time

            if "threat confirmed" in analysis.lower():
                self.threat_confirmation_count += 1

            if self.frames_analyzed == 3:
                self.analysis_complete = True
                final_assessment = self._make_final_assessment()
                return {"status": "final_assessment", "analysis": final_assessment}

            return {"status": "confirmation", "analysis": analysis}

        return {"status": "pending"}

    def _get_initial_system_prompt(self) -> str:
        return """You are a security surveillance AI analyzer specializing in weapon threat assessment. A weapon has been detected in this frame with high confidence (>75%). Your task is to provide detailed analysis focusing on the weapon threat.

        Analyze and report concisely in the following structure:
        
        THREAT: [Weapon details]
        SUSPECT: [Brief description - clothing, appearance and behaviour/Action]
        Risk Level: [High/Medium/Low]

        Keep total response under 100 words, focusing only on critical security details."""

    def _get_analysis_message(self, base64_image: str) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        return [
            {
                "type": "text",
                "text": "Analyze this security camera frame for potential threats. Provide a detailed threat assessment."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]

    def _get_confirmation_system_prompt(self) -> str:
        return """Compare with previous analysis. Provide brief update in format:

        Update: [Key changes in threat/location/behavior]

        Keep total response under 30 words."""

    def _get_confirmation_message(self, base64_image: str, previous_analysis: Dict) -> List[Dict]:
        return [
            {
                "type": "text",
                "text": f"Previous analysis: {previous_analysis}\n\nAnalyze this frame and confirm threat status."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]

    def _initial_threat_assessment(self, base64_image: str) -> str:
        """Perform initial detailed threat assessment"""
        messages = [
            SystemMessage(content=self._get_initial_system_prompt()),
            HumanMessage(content=self._get_analysis_message(base64_image))
        ]
        response = self.llm.invoke(messages)
        self.memory.save_context(
            {"input": "Initial threat assessment"},
            {"output": response.content}
        )
        return response.content

    def _confirmatory_analysis(self, base64_image: str) -> str:
        """Perform confirmatory analysis"""
        previous_analysis = self.memory.load_memory_variables({})

        messages = [
            SystemMessage(content=self._get_confirmation_system_prompt()),
            HumanMessage(content=self._get_confirmation_message(base64_image, previous_analysis))
        ]
        response = self.llm.invoke(messages)
        self.memory.save_context(
            {"input": f"Confirmation analysis {self.frames_analyzed}"},
            {"output": response.content}
        )
        return response.content

    def _make_final_assessment(self) -> Dict:
        """Make final threat assessment based on previous analyses"""
        all_analyses = self.memory.load_memory_variables({})

        # Extract risk levels from previous analyses
        risk_levels = []
        for analysis in all_analyses.values():
            if isinstance(analysis, str):
                # Look for Risk Level in the analysis text
                if "Risk Level: High" in analysis:
                    risk_levels.append("HIGH")
                elif "Risk Level: Medium" in analysis:
                    risk_levels.append("MEDIUM")
                elif "Risk Level: Low" in analysis:
                    risk_levels.append("LOW")

        # Determine final threat level based on risk assessments
        if "HIGH" in risk_levels:
            threat_level = "HIGH"
            recommendation = (
                f"CRITICAL: Armed threat confirmed at {self.location_info['location']}. "
                "Immediate security response required."
            )
        else:
            threat_level = "MEDIUM"
            recommendation = (
                f"ALERT: Potential threat at {self.location_info['location']}. "
                "Security verification needed."
            )

        return {
            "location": self.location_info['location'],
            "threat_level": threat_level,
            "risk_history": risk_levels,
            "final_recommendation": recommendation
        }