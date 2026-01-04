# Scorched Earth Modern - Sound System
# Procedural sound effects and music generation using pygame and numpy

import pygame
import numpy as np
import math
import random

from settings import (
    SOUND_ENABLED, MUSIC_ENABLED, SFX_VOLUME, MUSIC_VOLUME, SAMPLE_RATE
)


class SoundGenerator:
    """Generates sound effects and music procedurally using numpy waveforms."""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.sounds = {}
        self._generate_all_sounds()

    def _generate_all_sounds(self):
        """Pre-generate all sound effects."""
        # Weapon fire sounds
        self.sounds['fire_standard'] = self._generate_fire_sound(freq=200, duration=0.15)
        self.sounds['fire_big'] = self._generate_fire_sound(freq=120, duration=0.25)
        self.sounds['fire_nuke'] = self._generate_fire_sound(freq=80, duration=0.35)
        self.sounds['fire_dirt'] = self._generate_plop_sound()
        self.sounds['fire_mirv'] = self._generate_fire_sound(freq=300, duration=0.2)

        # Explosion sounds
        self.sounds['explosion_small'] = self._generate_explosion(duration=0.4, intensity=0.6)
        self.sounds['explosion_medium'] = self._generate_explosion(duration=0.6, intensity=0.8)
        self.sounds['explosion_large'] = self._generate_explosion(duration=0.9, intensity=1.0)
        self.sounds['explosion_nuke'] = self._generate_nuke_explosion()

        # Impact sounds
        self.sounds['tank_hit'] = self._generate_metal_hit()
        self.sounds['tank_destroy'] = self._generate_tank_destroy()

        # UI sounds
        self.sounds['turn_change'] = self._generate_beep(freq=440, duration=0.1)
        self.sounds['menu_select'] = self._generate_beep(freq=660, duration=0.08)
        self.sounds['menu_move'] = self._generate_beep(freq=330, duration=0.05)
        self.sounds['game_start'] = self._generate_fanfare()
        self.sounds['victory'] = self._generate_victory_fanfare()

        # Misc
        self.sounds['mirv_split'] = self._generate_split_sound()
        self.sounds['wind_change'] = self._generate_whoosh()
        self.sounds['power_adjust'] = self._generate_tick()

    def _make_sound(self, samples):
        """Convert numpy array to pygame Sound object."""
        # Ensure samples are in correct format (16-bit signed integers)
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
        # Make stereo
        stereo = np.column_stack((samples, samples))
        return pygame.sndarray.make_sound(stereo)

    def _generate_noise(self, duration):
        """Generate white noise."""
        num_samples = int(self.sample_rate * duration)
        return np.random.uniform(-1, 1, num_samples)

    def _generate_sine(self, freq, duration, amplitude=1.0):
        """Generate a sine wave."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        return amplitude * np.sin(2 * np.pi * freq * t)

    def _generate_square(self, freq, duration, amplitude=1.0):
        """Generate a square wave."""
        sine = self._generate_sine(freq, duration, amplitude)
        return np.sign(sine)

    def _generate_sawtooth(self, freq, duration, amplitude=1.0):
        """Generate a sawtooth wave."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        return amplitude * (2 * (t * freq - np.floor(t * freq + 0.5)))

    def _apply_envelope(self, samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
        """Apply ADSR envelope to samples."""
        num_samples = len(samples)
        envelope = np.ones(num_samples)

        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples

        if sustain_samples < 0:
            # Simplified envelope if duration is short
            envelope = np.linspace(1, 0, num_samples)
        else:
            # Attack
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            # Decay
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain, decay_samples)
            # Sustain
            envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain
            # Release
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)

        return samples * envelope

    def _apply_fade_out(self, samples, fade_duration=0.1):
        """Apply fade out to end of samples."""
        fade_samples = int(fade_duration * self.sample_rate)
        fade_samples = min(fade_samples, len(samples))
        if fade_samples > 0:
            fade = np.linspace(1, 0, fade_samples)
            samples[-fade_samples:] *= fade
        return samples

    def _lowpass_filter(self, samples, cutoff_ratio=0.1):
        """Simple lowpass filter using moving average."""
        window_size = max(1, int(1 / cutoff_ratio))
        kernel = np.ones(window_size) / window_size
        return np.convolve(samples, kernel, mode='same')

    def _generate_fire_sound(self, freq=200, duration=0.15):
        """Generate a cannon/weapon fire sound."""
        # Start with noise burst
        noise = self._generate_noise(duration) * 0.8

        # Add low frequency thump
        thump = self._generate_sine(freq, duration, 0.6)
        thump_decay = np.exp(-np.linspace(0, 10, len(thump)))
        thump *= thump_decay

        # Combine
        samples = noise + thump

        # Apply sharp attack, quick decay
        samples = self._apply_envelope(samples, attack=0.005, decay=0.05, sustain=0.3, release=0.05)

        # Lowpass to remove harsh high frequencies
        samples = self._lowpass_filter(samples, 0.15)

        return self._make_sound(samples * 0.7)

    def _generate_plop_sound(self):
        """Generate a dirt ball plop sound."""
        duration = 0.2

        # Descending tone
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        freq = 400 * np.exp(-t * 10)  # Rapidly descending frequency
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        samples = 0.5 * np.sin(phase)

        # Add some noise
        samples += self._generate_noise(duration) * 0.2

        samples = self._apply_envelope(samples, attack=0.01, decay=0.05, sustain=0.3, release=0.1)
        samples = self._lowpass_filter(samples, 0.2)

        return self._make_sound(samples * 0.6)

    def _generate_explosion(self, duration=0.5, intensity=1.0):
        """Generate an explosion sound."""
        # Noise-based explosion
        noise = self._generate_noise(duration)

        # Low frequency rumble
        rumble = self._generate_sine(60 * intensity, duration, 0.5)
        rumble += self._generate_sine(40 * intensity, duration, 0.3)

        # Combine with exponential decay
        decay = np.exp(-np.linspace(0, 8 / intensity, len(noise)))
        samples = (noise * 0.7 + rumble * 0.5) * decay

        # Initial pop/crack
        pop_duration = 0.02
        pop_samples = int(self.sample_rate * pop_duration)
        pop = self._generate_noise(pop_duration) * 1.5
        samples[:pop_samples] += pop * np.exp(-np.linspace(0, 5, pop_samples))

        # Lowpass filter
        samples = self._lowpass_filter(samples, 0.1 + intensity * 0.1)

        return self._make_sound(samples * 0.8)

    def _generate_nuke_explosion(self):
        """Generate a massive nuclear explosion sound."""
        duration = 1.5

        # Multiple layers of noise
        noise = self._generate_noise(duration)

        # Very low frequency rumble
        rumble = self._generate_sine(30, duration, 0.4)
        rumble += self._generate_sine(20, duration, 0.3)
        rumble += self._generate_sine(50, duration, 0.2)

        # Slower decay for bigger explosion
        decay = np.exp(-np.linspace(0, 4, len(noise)))

        # Build up then decay
        num_samples = len(noise)
        buildup = np.ones(num_samples)
        buildup_samples = int(0.1 * self.sample_rate)
        buildup[:buildup_samples] = np.linspace(0.3, 1.0, buildup_samples)

        samples = (noise * 0.6 + rumble * 0.6) * decay * buildup

        # Big initial crack
        pop_duration = 0.05
        pop_samples = int(self.sample_rate * pop_duration)
        pop = self._generate_noise(pop_duration) * 2.0
        samples[:pop_samples] += pop * np.exp(-np.linspace(0, 3, pop_samples))

        samples = self._lowpass_filter(samples, 0.15)

        return self._make_sound(samples * 0.9)

    def _generate_metal_hit(self):
        """Generate a metallic hit sound for tank damage."""
        duration = 0.3

        # Multiple resonant frequencies for metallic sound
        samples = np.zeros(int(self.sample_rate * duration))

        # Metallic frequencies
        freqs = [800, 1200, 1600, 2400]
        for i, freq in enumerate(freqs):
            tone = self._generate_sine(freq + random.randint(-50, 50), duration, 0.3 / (i + 1))
            decay = np.exp(-np.linspace(0, 15 + i * 5, len(tone)))
            samples += tone * decay

        # Add impact noise
        noise = self._generate_noise(0.05) * 0.5
        samples[:len(noise)] += noise * np.exp(-np.linspace(0, 10, len(noise)))

        return self._make_sound(samples * 0.6)

    def _generate_tank_destroy(self):
        """Generate tank destruction sound."""
        duration = 0.8

        # Combination of explosion and metal
        explosion = self._generate_noise(duration) * 0.6

        # Crunching metal
        metal_freqs = [200, 400, 600, 150]
        metal = np.zeros(int(self.sample_rate * duration))
        for freq in metal_freqs:
            tone = self._generate_sine(freq, duration, 0.2)
            decay = np.exp(-np.linspace(0, 8, len(tone)))
            metal += tone * decay

        # Combine with decay
        decay = np.exp(-np.linspace(0, 5, len(explosion)))
        samples = (explosion + metal) * decay

        samples = self._lowpass_filter(samples, 0.2)

        return self._make_sound(samples * 0.8)

    def _generate_beep(self, freq=440, duration=0.1):
        """Generate a simple beep."""
        samples = self._generate_sine(freq, duration, 0.5)
        samples = self._apply_envelope(samples, attack=0.01, decay=0.02, sustain=0.8, release=0.02)
        return self._make_sound(samples * 0.4)

    def _generate_fanfare(self):
        """Generate game start fanfare."""
        duration_per_note = 0.15
        notes = [262, 330, 392, 523]  # C4, E4, G4, C5

        all_samples = []
        for note in notes:
            samples = self._generate_sine(note, duration_per_note, 0.5)
            samples += self._generate_sine(note * 2, duration_per_note, 0.2)  # Octave harmonic
            samples = self._apply_envelope(samples, attack=0.01, decay=0.05, sustain=0.7, release=0.05)
            all_samples.extend(samples)

        return self._make_sound(np.array(all_samples) * 0.5)

    def _generate_victory_fanfare(self):
        """Generate victory fanfare."""
        duration_per_note = 0.2
        # Triumphant chord progression
        chords = [
            [262, 330, 392],  # C major
            [294, 370, 440],  # D major
            [330, 415, 494],  # E major
            [392, 494, 587],  # G major
        ]

        all_samples = []
        for chord in chords:
            samples = np.zeros(int(self.sample_rate * duration_per_note))
            for note in chord:
                tone = self._generate_sine(note, duration_per_note, 0.3)
                tone += self._generate_sine(note * 2, duration_per_note, 0.1)
                samples += tone
            samples = self._apply_envelope(samples, attack=0.02, decay=0.05, sustain=0.7, release=0.05)
            all_samples.extend(samples)

        # Hold last chord longer
        hold_duration = 0.5
        last_chord = chords[-1]
        samples = np.zeros(int(self.sample_rate * hold_duration))
        for note in last_chord:
            samples += self._generate_sine(note, hold_duration, 0.3)
            samples += self._generate_sine(note * 2, hold_duration, 0.15)
        samples = self._apply_envelope(samples, attack=0.01, decay=0.1, sustain=0.8, release=0.2)
        all_samples.extend(samples)

        return self._make_sound(np.array(all_samples) * 0.5)

    def _generate_split_sound(self):
        """Generate MIRV split sound."""
        duration = 0.2

        # Rising pitch burst
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        freq = 300 + 1000 * t  # Rising frequency
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        samples = 0.4 * np.sin(phase)

        # Add noise burst
        samples += self._generate_noise(duration) * 0.3

        samples = self._apply_envelope(samples, attack=0.01, decay=0.05, sustain=0.5, release=0.1)

        return self._make_sound(samples * 0.5)

    def _generate_whoosh(self):
        """Generate wind whoosh sound."""
        duration = 0.3

        noise = self._generate_noise(duration)

        # Band-pass effect through filtering
        samples = self._lowpass_filter(noise, 0.3)

        # Volume envelope
        num_samples = len(samples)
        envelope = np.sin(np.linspace(0, np.pi, num_samples))
        samples *= envelope

        return self._make_sound(samples * 0.3)

    def _generate_tick(self):
        """Generate a subtle tick sound for power adjustment."""
        duration = 0.02
        samples = self._generate_sine(1000, duration, 0.3)
        samples = self._apply_fade_out(samples, 0.01)
        return self._make_sound(samples * 0.2)

    def get_sound(self, name):
        """Get a pre-generated sound by name."""
        return self.sounds.get(name)


class MusicGenerator:
    """Generates procedural background music."""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.current_music = None
        self.music_channel = None

    def _generate_sine(self, freq, duration, amplitude=1.0):
        """Generate a sine wave."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        return amplitude * np.sin(2 * np.pi * freq * t)

    def _apply_envelope(self, samples, attack=0.05, release=0.1):
        """Apply simple attack-release envelope."""
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)

        if attack_samples > 0 and attack_samples < len(samples):
            samples[:attack_samples] *= np.linspace(0, 1, attack_samples)
        if release_samples > 0 and release_samples < len(samples):
            samples[-release_samples:] *= np.linspace(1, 0, release_samples)

        return samples

    def _note_to_freq(self, note):
        """Convert MIDI note number to frequency."""
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    def generate_menu_music(self, duration=30.0):
        """Generate ambient menu music."""
        num_samples = int(self.sample_rate * duration)
        samples = np.zeros(num_samples)

        # Ambient pad with slow chord changes
        # C minor pentatonic: C, Eb, F, G, Bb
        scale = [48, 51, 53, 55, 58, 60, 63, 65, 67, 70]  # MIDI notes

        # Slow evolving pad
        chord_duration = 4.0
        num_chords = int(duration / chord_duration)

        for i in range(num_chords):
            start_sample = int(i * chord_duration * self.sample_rate)
            end_sample = int((i + 1) * chord_duration * self.sample_rate)
            chord_samples = end_sample - start_sample

            # Pick 3 notes for chord
            chord_notes = random.sample(scale, 3)

            for note in chord_notes:
                freq = self._note_to_freq(note)
                t = np.linspace(0, chord_duration, chord_samples, False)

                # Soft sine with slight detuning for warmth
                tone = 0.1 * np.sin(2 * np.pi * freq * t)
                tone += 0.05 * np.sin(2 * np.pi * freq * 1.002 * t)  # Slight detune
                tone += 0.03 * np.sin(2 * np.pi * freq * 2 * t)  # Octave harmonic

                # Soft envelope
                attack = int(0.5 * self.sample_rate)
                release = int(0.5 * self.sample_rate)
                envelope = np.ones(chord_samples)
                if attack < chord_samples:
                    envelope[:attack] = np.linspace(0, 1, attack)
                if release < chord_samples:
                    envelope[-release:] = np.linspace(1, 0, release)

                tone *= envelope

                if start_sample + len(tone) <= num_samples:
                    samples[start_sample:start_sample + len(tone)] += tone

        # Add subtle low drone
        drone_freq = self._note_to_freq(36)  # C2
        t = np.linspace(0, duration, num_samples, False)
        drone = 0.05 * np.sin(2 * np.pi * drone_freq * t)
        drone += 0.03 * np.sin(2 * np.pi * drone_freq * 2 * t)
        samples += drone

        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val * 0.5

        # Convert to pygame sound
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))

        return pygame.sndarray.make_sound(stereo)

    def generate_battle_music(self, duration=60.0):
        """Generate more intense battle music."""
        num_samples = int(self.sample_rate * duration)
        samples = np.zeros(num_samples)

        # More aggressive scale: D minor
        bass_notes = [38, 41, 43, 45]  # D, F, G, A (low)

        # Bass pulse pattern
        beat_duration = 0.5  # 120 BPM
        num_beats = int(duration / beat_duration)

        for i in range(num_beats):
            start_sample = int(i * beat_duration * self.sample_rate)
            note = bass_notes[i % len(bass_notes)]
            freq = self._note_to_freq(note)

            note_duration = 0.3
            note_samples = int(note_duration * self.sample_rate)
            t = np.linspace(0, note_duration, note_samples, False)

            # Punchy bass
            tone = 0.15 * np.sin(2 * np.pi * freq * t)
            tone += 0.08 * np.sin(2 * np.pi * freq * 2 * t)

            # Quick decay
            decay = np.exp(-t * 5)
            tone *= decay

            if start_sample + len(tone) <= num_samples:
                samples[start_sample:start_sample + len(tone)] += tone

        # Add pad layer (slower)
        pad_notes = [50, 53, 57, 62]  # D, F, A, D (higher)
        pad_duration = 2.0
        num_pads = int(duration / pad_duration)

        for i in range(num_pads):
            start_sample = int(i * pad_duration * self.sample_rate)
            note = pad_notes[i % len(pad_notes)]
            freq = self._note_to_freq(note)

            pad_samples = int(pad_duration * self.sample_rate)
            t = np.linspace(0, pad_duration, pad_samples, False)

            tone = 0.06 * np.sin(2 * np.pi * freq * t)
            tone += 0.04 * np.sin(2 * np.pi * freq * 1.5 * t)  # Fifth

            # Soft envelope
            attack = int(0.2 * self.sample_rate)
            release = int(0.3 * self.sample_rate)
            envelope = np.ones(pad_samples)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            tone *= envelope

            if start_sample + len(tone) <= num_samples:
                samples[start_sample:start_sample + len(tone)] += tone

        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val * 0.4

        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
        stereo = np.column_stack((samples, samples))

        return pygame.sndarray.make_sound(stereo)


class SoundManager:
    """Manages all game audio: sound effects and music."""

    def __init__(self):
        self.enabled = SOUND_ENABLED
        self.music_enabled = MUSIC_ENABLED
        self.sfx_volume = SFX_VOLUME
        self.music_volume = MUSIC_VOLUME

        # Initialize sound generator with configured sample rate
        self.generator = SoundGenerator(sample_rate=SAMPLE_RATE)
        self.music_generator = MusicGenerator(sample_rate=SAMPLE_RATE)

        # Music state
        self.current_music = None
        self.music_channel = None

        # Reserve channel for music
        pygame.mixer.set_num_channels(16)
        self.music_channel = pygame.mixer.Channel(15)

    def set_sfx_volume(self, volume):
        """Set sound effects volume (0.0 to 1.0)."""
        self.sfx_volume = max(0.0, min(1.0, volume))

    def set_music_volume(self, volume):
        """Set music volume (0.0 to 1.0)."""
        self.music_volume = max(0.0, min(1.0, volume))
        if self.music_channel:
            self.music_channel.set_volume(self.music_volume)

    def toggle_sound(self):
        """Toggle all sound on/off."""
        self.enabled = not self.enabled
        if not self.enabled:
            pygame.mixer.stop()
        return self.enabled

    def toggle_music(self):
        """Toggle music on/off."""
        self.music_enabled = not self.music_enabled
        if not self.music_enabled:
            if self.music_channel:
                self.music_channel.stop()
        return self.music_enabled

    def play(self, sound_name, volume_multiplier=1.0):
        """Play a sound effect by name."""
        if not self.enabled:
            return

        sound = self.generator.get_sound(sound_name)
        if sound:
            sound.set_volume(self.sfx_volume * volume_multiplier)
            sound.play()

    def play_fire(self, weapon_name):
        """Play firing sound based on weapon."""
        weapon_sounds = {
            'Standard Shell': 'fire_standard',
            'Big Bertha': 'fire_big',
            'Baby Nuke': 'fire_nuke',
            'Dirt Ball': 'fire_dirt',
            'MIRV': 'fire_mirv',
        }
        sound_name = weapon_sounds.get(weapon_name, 'fire_standard')
        self.play(sound_name)

    def play_explosion(self, radius):
        """Play explosion sound based on explosion radius."""
        if radius <= 35:
            self.play('explosion_small')
        elif radius <= 55:
            self.play('explosion_medium')
        elif radius <= 75:
            self.play('explosion_large')
        else:
            self.play('explosion_nuke')

    def play_tank_hit(self, destroyed=False):
        """Play tank hit or destruction sound."""
        if destroyed:
            self.play('tank_destroy')
        else:
            self.play('tank_hit')

    def play_menu_music(self):
        """Start playing menu music."""
        if not self.enabled or not self.music_enabled:
            return

        if self.current_music != 'menu':
            self.current_music = 'menu'
            music = self.music_generator.generate_menu_music(duration=30.0)
            self.music_channel.set_volume(self.music_volume)
            self.music_channel.play(music, loops=-1)

    def play_battle_music(self):
        """Start playing battle music."""
        if not self.enabled or not self.music_enabled:
            return

        if self.current_music != 'battle':
            self.current_music = 'battle'
            music = self.music_generator.generate_battle_music(duration=60.0)
            self.music_channel.set_volume(self.music_volume)
            self.music_channel.play(music, loops=-1)

    def stop_music(self):
        """Stop current music."""
        if self.music_channel:
            self.music_channel.stop()
        self.current_music = None

    def fade_out_music(self, time_ms=1000):
        """Fade out current music."""
        if self.music_channel:
            self.music_channel.fadeout(time_ms)
        self.current_music = None
