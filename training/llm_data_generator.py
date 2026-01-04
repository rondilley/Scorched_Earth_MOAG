# LLM-based training data generator
# Uses LLM API to generate training data OFFLINE (not during gameplay)

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from training.config import LLM_CONFIG, ACTION_CONFIG


def load_api_key(provider: str) -> Optional[str]:
    """Load API key from *.key.txt file.

    Searches in current directory, training directory, and home directory.

    Args:
        provider: 'anthropic' or 'openai'

    Returns:
        API key string or None if not found
    """
    key_filename = f"{provider}.key.txt"

    search_paths = [
        '.',  # Current directory
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Project root
        os.path.dirname(os.path.abspath(__file__)),  # Training directory
        os.path.expanduser('~'),  # Home directory
    ]

    for search_path in search_paths:
        key_path = os.path.join(search_path, key_filename)
        if os.path.exists(key_path):
            try:
                with open(key_path, 'r') as f:
                    key = f.read().strip()
                    if key:
                        return key
            except IOError:
                continue

    return None


def format_game_state_for_llm(obs: Dict) -> str:
    """Format observation as text for LLM prompt.

    Args:
        obs: Observation dict from environment

    Returns:
        Formatted text description of game state
    """
    # Extract player info
    player = obs['player_tank']
    player_x = int(player[0] * 1024)  # Denormalize
    player_y = int(player[1] * 768)
    player_health = int(player[2] * 100)
    player_angle = int(player[3] * 180)
    player_power = int(player[4] * 450 + 50)

    # Extract opponent info
    opponents = obs['opponent_tanks']
    opponent_texts = []
    for i in range(0, len(opponents), 4):
        if i + 3 < len(opponents):
            ox = int(opponents[i] * 1024)
            oy = int(opponents[i+1] * 768)
            oh = int(opponents[i+2] * 100)
            alive = "alive" if opponents[i+3] > 0.5 else "dead"
            if oh > 0 or alive == "dead":
                opponent_texts.append(f"  - Enemy at ({ox}, {oy}), health: {oh}, status: {alive}")

    # Wind
    wind = obs['wind'][0]
    wind_dir = "right" if wind > 0 else "left"
    wind_strength = abs(wind) * 8  # Denormalize

    # Terrain summary (sample key points)
    terrain = obs['terrain']
    terrain_samples = []
    for x in [100, 300, 500, 700, 900]:
        if x < len(terrain):
            h = int(terrain[x] * 500)  # Approximate height
            terrain_samples.append(f"x={x}: height={h}")

    state_text = f"""GAME STATE:
Your Tank:
  - Position: ({player_x}, {player_y})
  - Health: {player_health}/100
  - Current angle: {player_angle} degrees (0=right, 90=up, 180=left)
  - Current power: {player_power} (range: 50-500)

Enemies:
{chr(10).join(opponent_texts) if opponent_texts else "  - No enemies visible"}

Wind: {wind_strength:.1f} pixels/sec to the {wind_dir}

Terrain (sample heights):
  {', '.join(terrain_samples)}

WEAPONS:
0: Standard Shell - 25 damage, 30 radius
1: Big Bertha - 40 damage, 50 radius
2: Baby Nuke - 75 damage, 80 radius
3: Dirt Ball - 0 damage, adds terrain
4: MIRV - 20 damage each, splits into 5 projectiles"""

    return state_text


class LLMDataGenerator:
    """Generates training data using LLM API calls.

    This is for OFFLINE data generation only - not real-time gameplay.
    Use this to:
    - Generate expert demonstrations
    - Create training examples
    - Develop reward shaping strategies
    """

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        temperature: float = None,
        timeout: float = None,
        max_retries: int = None
    ):
        """Initialize LLM data generator.

        Args:
            provider: 'anthropic' or 'openai'
            model: Model name (uses default from config if None)
            temperature: Sampling temperature
            timeout: API timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.provider = provider or LLM_CONFIG['default_provider']
        self.model = model or LLM_CONFIG['models'].get(self.provider)
        self.temperature = temperature or LLM_CONFIG['temperature']
        self.timeout = timeout or LLM_CONFIG['timeout']
        self.max_retries = max_retries or LLM_CONFIG['max_retries']

        # Load API key
        self.api_key = load_api_key(self.provider)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Please create {self.provider}.key.txt "
                f"in the project directory with your API key."
            )

        # Initialize client
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the API client."""
        if self.provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        elif self.provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_api(self, prompt: str, system_prompt: str = None) -> str:
        """Make API call with retries.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Response text
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == 'anthropic':
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                        system=system_prompt or "You are an expert game AI strategist.",
                    )
                    return message.content[0].text

                elif self.provider == 'openai':
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1024,
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

        return ""

    def generate_action_for_state(self, obs: Dict) -> Tuple[Dict, str]:
        """Generate expert action for a game state.

        Args:
            obs: Observation dict from environment

        Returns:
            action: Dict with angle, power, weapon
            reasoning: Explanation of the decision
        """
        state_text = format_game_state_for_llm(obs)

        prompt = f"""{state_text}

Analyze this game state and decide the best action. Consider:
1. Distance and direction to enemies
2. Wind compensation needed
3. Terrain obstacles
4. Weapon selection based on situation

Respond with a JSON object:
{{"angle": <0-180>, "power": <50-500>, "weapon": <0-4>, "reasoning": "<explanation>"}}

Only output the JSON, no other text."""

        system = "You are an expert Scorched Earth player. Analyze the game state and provide optimal firing solutions."

        response = self._call_api(prompt, system)

        # Parse response
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                action = {
                    'angle': float(np.clip(data.get('angle', 45), 0, 180)),
                    'power': float(np.clip(data.get('power', 300), 50, 500)),
                    'weapon': int(np.clip(data.get('weapon', 0), 0, 4)),
                }
                reasoning = data.get('reasoning', '')
                return action, reasoning
        except json.JSONDecodeError:
            pass

        # Fallback action
        return {'angle': 45.0, 'power': 300.0, 'weapon': 0}, "Failed to parse response"

    def generate_training_batch(
        self,
        env,
        num_examples: int = 100,
        save_path: str = None
    ) -> List[Dict]:
        """Generate a batch of training examples.

        Args:
            env: ScorchedEarthEnv instance
            num_examples: Number of examples to generate
            save_path: Optional path to save examples as JSON

        Returns:
            List of training examples
        """
        import sys

        print("\n" + "=" * 60)
        print("LLM Training Data Generation")
        print("=" * 60)
        print(f"  Provider:     {self.provider}")
        print(f"  Model:        {self.model}")
        print(f"  Examples:     {num_examples}")
        if save_path:
            print(f"  Output file:  {save_path}")
        print("=" * 60 + "\n")
        sys.stdout.flush()

        examples = []
        start_time = time.time()
        errors = 0

        for i in range(num_examples):
            example_start = time.time()

            # Reset to get a fresh state
            obs, info = env.reset()

            # Generate expert action
            try:
                action, reasoning = self.generate_action_for_state(obs)
            except Exception as e:
                errors += 1
                print(f"  [ERROR] Example {i+1}: {e}")
                # Use fallback action
                action = {'angle': 45.0, 'power': 300.0, 'weapon': 0}
                reasoning = f"Error: {e}"

            example = {
                'observation': {
                    'terrain': obs['terrain'].tolist(),
                    'player_tank': obs['player_tank'].tolist(),
                    'opponent_tanks': obs['opponent_tanks'].tolist(),
                    'wind': obs['wind'].tolist(),
                },
                'action': action,
                'reasoning': reasoning,
            }
            examples.append(example)

            # Progress output
            elapsed = time.time() - start_time
            example_time = time.time() - example_start
            avg_time = elapsed / (i + 1)
            eta = avg_time * (num_examples - i - 1)
            progress = (i + 1) / num_examples * 100

            print(f"[{progress:5.1f}%] Example {i+1}/{num_examples} | "
                  f"Time: {example_time:.1f}s | ETA: {eta:.0f}s | "
                  f"Action: angle={action['angle']:.0f}, power={action['power']:.0f}, weapon={action['weapon']}")
            sys.stdout.flush()

            # Small delay to avoid rate limits
            time.sleep(0.5)

        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"  Examples generated: {len(examples)}")
        print(f"  Errors:             {errors}")
        print(f"  Total time:         {total_time:.1f}s")
        print(f"  Avg time/example:   {total_time/num_examples:.1f}s")

        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(examples, f, indent=2)
                print(f"  Saved to:           {save_path}")
            except IOError as e:
                print(f"  [ERROR] Failed to save: {e}")

        print("=" * 60 + "\n")

        return examples

    def generate_reward_shaping_suggestions(self, game_description: str = None) -> Dict:
        """Ask LLM for reward shaping suggestions.

        Args:
            game_description: Optional description of game mechanics

        Returns:
            Dict with reward component suggestions
        """
        prompt = """For a Scorched Earth artillery game, suggest reward shaping for training an RL agent.

Current reward structure:
- +0.1 per damage dealt
- +5.0 per kill
- +20.0 for winning
- -10.0 for losing
- -0.05 per self-damage
- -0.01 per step

Suggest improvements or additional reward components. Consider:
1. Exploration bonuses
2. Accuracy rewards
3. Strategic positioning
4. Resource management

Respond with a JSON object containing suggested reward components and their weights."""

        response = self._call_api(prompt)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return {'error': 'Failed to parse suggestions'}

    def generate_curriculum_stages(self) -> List[Dict]:
        """Generate curriculum learning stages.

        Returns:
            List of training stage configurations
        """
        prompt = """Design a curriculum learning strategy for training an RL agent to play Scorched Earth.

The agent needs to learn:
1. Basic aiming (angle/power)
2. Wind compensation
3. Weapon selection
4. Strategic positioning

Design 4-5 training stages with increasing difficulty. For each stage, specify:
- Stage name
- Number of opponents
- Opponent difficulty (easy/medium/hard)
- Training steps
- Success criteria

Respond with a JSON array of stage objects."""

        response = self._call_api(prompt)

        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        # Default curriculum
        return [
            {'name': 'Basic Aiming', 'opponents': 1, 'difficulty': 'easy', 'steps': 50000},
            {'name': 'Wind Compensation', 'opponents': 1, 'difficulty': 'medium', 'steps': 100000},
            {'name': 'Multi-Enemy', 'opponents': 2, 'difficulty': 'medium', 'steps': 150000},
            {'name': 'Full Game', 'opponents': 3, 'difficulty': 'hard', 'steps': 200000},
        ]


def load_training_examples(path: str) -> List[Dict]:
    """Load training examples from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        List of training examples
    """
    with open(path, 'r') as f:
        return json.load(f)


def examples_to_dataset(examples: List[Dict], device='cpu'):
    """Convert training examples to tensor dataset.

    Args:
        examples: List of training examples
        device: Torch device

    Returns:
        Tuple of (observations, actions) tensor dicts
    """
    import torch

    observations = {
        'terrain': torch.tensor(
            [e['observation']['terrain'] for e in examples],
            dtype=torch.float32, device=device
        ),
        'player_tank': torch.tensor(
            [e['observation']['player_tank'] for e in examples],
            dtype=torch.float32, device=device
        ),
        'opponent_tanks': torch.tensor(
            [e['observation']['opponent_tanks'] for e in examples],
            dtype=torch.float32, device=device
        ),
        'wind': torch.tensor(
            [e['observation']['wind'] for e in examples],
            dtype=torch.float32, device=device
        ),
    }

    actions = {
        'angle': torch.tensor(
            [e['action']['angle'] for e in examples],
            dtype=torch.float32, device=device
        ),
        'power': torch.tensor(
            [e['action']['power'] for e in examples],
            dtype=torch.float32, device=device
        ),
        'weapon': torch.tensor(
            [e['action']['weapon'] for e in examples],
            dtype=torch.long, device=device
        ),
    }

    return observations, actions
