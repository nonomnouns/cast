import os
import json
import random
import asyncio
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from farcaster import Warpcast
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class CastManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.load_configs()
        
        # Enhanced state management
        self.daily_cast_count = 0
        self.last_cast_time = None
        self.current_personality = None  # Will be set during async setup
        self.current_topic = None
        self.engagement_data = []
        self.tuned_model = None
        self.client = None  # Will be set during async setup
        
        # Optimized posting hours (peak engagement times)
        self.optimal_hours = {
            'weekday': [(9, 11), (13, 15), (19, 22)],  # Peak hours during weekdays
            'weekend': [(11, 14), (15, 18), (20, 23)]   # Peak hours during weekends
        }

    def load_configs(self):
        """Load configuration files"""
        try:
            with open('0tto.json', 'r') as f:
                self.otto_config = json.load(f)
            with open('config.json', 'r') as f:
                self.tech_config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configs: {e}")
            raise

    async def initialize(self):
        """Async initialization method"""
        try:
            await self.setup_clients()
            self.current_personality = random.choice(self.otto_config["bio"])
            self.logger.info("CastManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CastManager: {e}")
            raise

    async def setup_clients(self):
        """Async setup of clients"""
        try:
            # Farcaster client
            mnemonic = os.getenv("FARCASTER_MNEMONIC")
            if not mnemonic:
                raise ValueError("FARCASTER_MNEMONIC not found in environment")
            self.client = Warpcast(mnemonic=mnemonic)
            
            # Gemini client setup
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            
            # Initialize tuned model
            await self.setup_tuned_model()
            
        except Exception as e:
            self.logger.error(f"Error setting up clients: {e}")
            raise

    async def setup_tuned_model(self):
        """Setup and configure the language model"""
        try:
            # Check for existing tuned models
            tuned_models = list(genai.list_tuned_models())
            model_name = "otto-personality-model"
            
            if not any(m.name.endswith(model_name) for m in tuned_models):
                # Create new tuned model if doesn't exist
                await self.create_tuned_model()
            else:
                # Use existing model
                self.tuned_model = genai.GenerativeModel(
                    model_name=f"tunedModels/{model_name}"
                )
        except Exception as e:
            self.logger.error(f"Error setting up tuned model: {e}")
            # Fallback to base model
            self.setup_base_model()

    def setup_base_model(self):
        """Fallback to base model if tuning fails"""
        generation_config = genai.GenerationConfig(
            max_output_tokens=20,
            temperature=0.7,
            top_p=0.9,
            top_k=40
        )
        self.tuned_model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)

    async def create_tuned_model(self):
        """Create and train a new tuned model"""
        try:
            # Prepare training data from successful posts
            training_data = self.prepare_training_data()
            
            # Create tuned model
            operation = genai.create_tuned_model(
                display_name="otto-personality-model",
                source_model="models/gemini-1.5-flash-001-tuning",
                epoch_count=self.tech_config["model_tuning"]["epoch_count"],
                batch_size=self.tech_config["model_tuning"]["batch_size"],
                learning_rate=self.tech_config["model_tuning"]["learning_rate"],
                training_data=training_data
            )
            
            # Wait for tuning to complete
            for status in operation.wait_bar():
                await asyncio.sleep(10)
            
            result = operation.result()
            self.tuned_model = genai.GenerativeModel(model_name=result.name)
            
        except Exception as e:
            self.logger.error(f"Error creating tuned model: {e}")
            self.setup_base_model()

    def prepare_training_data(self) -> List[Dict]:
        """Prepare training data for model tuning"""
        training_data = []
        for post in self.engagement_data:
            if post.get('engagement_score', 0) > self.tech_config['monitoring']['engagement_threshold']['likes']:
                training_data.append({
                    "text_input": post.get('prompt', ''),
                    "output": post.get('content', '')
                })
        return training_data

    def select_topic(self) -> str:
        """Select a random topic from configured topics"""
        return random.choice(self.otto_config["topics"])

    def create_prompt_context(self, topic: str) -> str:
        """Create context for cast generation"""
        personality = self.current_personality or random.choice(self.otto_config["bio"])
        template = self.otto_config["prompts"]["cast_template"]
        
        context = f"""
        Acting as {personality}
        {template.format(topic=topic)}
        
        Style guidelines:
        {' '.join(self.otto_config["style"]["post"])}
        
        Keep the response under 280 characters.
        """
        return context

    async def is_optimal_posting_time(self) -> bool:
        """Check if current time is optimal for posting"""
        current_time = datetime.now()
        is_weekend = current_time.weekday() >= 5
        current_hour = current_time.hour
        
        optimal_ranges = self.optimal_hours['weekend' if is_weekend else 'weekday']
        
        return any(start <= current_hour <= end for start, end in optimal_ranges)

    async def should_post(self) -> bool:
        """Check if we should post based on constraints"""
        if self.daily_cast_count >= self.tech_config["posting"]["max_daily_casts"]:
            return False
            
        if self.last_cast_time:
            time_since_last = datetime.now() - self.last_cast_time
            min_interval = self.tech_config["posting"]["cast_interval"]["min"]
            if time_since_last.total_seconds() < min_interval:
                return False
                
        return True

    async def generate_cast(self, topic: Optional[str] = None) -> str:
        """Generate a cast using the AI model"""
        try:
            self.current_topic = topic or self.select_topic()
            context = self.create_prompt_context(self.current_topic)
            
            # Use tuned model if available
            response = self.tuned_model.generate_content(context)
            cast_text = response.text.strip()
            
            if len(cast_text) > 280:
                cast_text = cast_text[:277] + "..."
                
            return cast_text
        except Exception as e:
            self.logger.error(f"Error generating cast: {e}")
            return ""

    async def post_cast(self, cast_text: str) -> bool:
        """Post a cast to Farcaster"""
        try:
            if not cast_text:
                return False
                
            response = await self.client.post_cast(text=cast_text)
            self.daily_cast_count += 1
            self.last_cast_time = datetime.now()
            
            # Start tracking engagement
            asyncio.create_task(self.track_engagement(response.hash))
            
            return True
        except Exception as e:
            self.logger.error(f"Error posting cast: {e}")
            return False

    async def track_engagement(self, cast_id: str):
        """Track engagement metrics for posts"""
        try:
            # Get cast metrics after some time
            await asyncio.sleep(3600)  # Wait 1 hour
            cast_data = await self.client.get_cast(cast_id)
            
            engagement_score = (
                cast_data.likes * 2 +  # Weight likes more heavily
                cast_data.recasts * 3   # Weight recasts most heavily
            )
            
            self.engagement_data.append({
                'cast_id': cast_id,
                'engagement_score': engagement_score,
                'timestamp': datetime.now(),
                'hour': datetime.now().hour
            })
            
            # Update optimal posting times based on engagement
            await self.update_optimal_times()
            
        except Exception as e:
            self.logger.error(f"Error tracking engagement: {e}")

    async def update_optimal_times(self):
        """Update optimal posting times based on engagement data"""
        if len(self.engagement_data) < 10:  # Need minimum data
            return
            
        # Group engagement by hour
        hour_engagement = {}
        for data in self.engagement_data:
            hour = data['hour']
            if hour not in hour_engagement:
                hour_engagement[hour] = []
            hour_engagement[hour].append(data['engagement_score'])
        
        # Calculate average engagement per hour
        avg_engagement = {
            hour: sum(scores)/len(scores) 
            for hour, scores in hour_engagement.items()
        }
        
        # Update optimal hours based on top performing times
        sorted_hours = sorted(
            avg_engagement.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top 6 hours and group them into ranges
        top_hours = [hour for hour, _ in sorted_hours[:6]]
        
        # Update optimal hours
        self.optimal_hours = {
            'weekday': [(h, h+2) for h in top_hours[:3]],
            'weekend': [(h, h+2) for h in top_hours[3:]]
        }

    def rotate_personality(self):
        """Rotate the bot's personality"""
        self.current_personality = random.choice(self.otto_config["bio"])

    async def auto_cast(self):
        """Enhanced main loop for automated casting"""
        self.logger.info("Starting auto cast loop...")
        
        while True:
            try:
                if await self.should_post() and await self.is_optimal_posting_time():
                    # Rotate personality occasionally
                    if random.random() < self.tech_config["optimization"]["personality_rotation"]["probability"]:
                        self.rotate_personality()
                    
                    # Generate and post cast
                    cast_text = await self.generate_cast()
                    if cast_text:
                        success = await self.post_cast(cast_text)
                        if success:
                            # Track engagement for successful posts
                            asyncio.create_task(self.track_engagement(cast_text))
                    
                    # Periodic model retraining
                    if len(self.engagement_data) >= self.tech_config["model_tuning"]["min_training_samples"]:
                        await self.create_tuned_model()
                        self.engagement_data = []  # Reset after retraining
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in auto cast loop: {e}")
                await asyncio.sleep(300)

async def main():
    """Entry point for running the cast manager directly"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = CastManager()
    await manager.initialize()
    await manager.auto_cast()

if __name__ == "__main__":
    asyncio.run(main())