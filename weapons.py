"""Weapon definitions and types."""

from enum import Enum, auto


class WeaponType(Enum):
    """Types of weapons available."""
    STANDARD = auto()
    BIG_BERTHA = auto()
    BABY_NUKE = auto()
    DIRT_BALL = auto()
    MIRV = auto()


class Weapon:
    """Base weapon class."""

    def __init__(self, name, weapon_type, damage, explosion_radius,
                 color=(255, 255, 255), special=None):
        """Initialize weapon.

        Args:
            name: Display name
            weapon_type: WeaponType enum value
            damage: Maximum damage at explosion center
            explosion_radius: Radius of explosion
            color: Projectile color
            special: Special behavior ('add_terrain', 'mirv', etc.)
        """
        self.name = name
        self.weapon_type = weapon_type
        self.damage = damage
        self.explosion_radius = explosion_radius
        self.color = color
        self.special = special

    def __repr__(self):
        return f"Weapon({self.name})"


# Define all weapons
WEAPONS = [
    Weapon(
        name="Standard Shell",
        weapon_type=WeaponType.STANDARD,
        damage=25,
        explosion_radius=30,
        color=(255, 255, 200)
    ),
    Weapon(
        name="Big Bertha",
        weapon_type=WeaponType.BIG_BERTHA,
        damage=40,
        explosion_radius=50,
        color=(255, 150, 100)
    ),
    Weapon(
        name="Baby Nuke",
        weapon_type=WeaponType.BABY_NUKE,
        damage=75,
        explosion_radius=80,
        color=(100, 255, 100)
    ),
    Weapon(
        name="Dirt Ball",
        weapon_type=WeaponType.DIRT_BALL,
        damage=0,
        explosion_radius=40,
        color=(139, 90, 43),
        special='add_terrain'
    ),
    Weapon(
        name="MIRV",
        weapon_type=WeaponType.MIRV,
        damage=20,
        explosion_radius=25,
        color=(255, 100, 255),
        special='mirv'
    ),
]


def get_weapon(index):
    """Get weapon by index (wraps around)."""
    return WEAPONS[index % len(WEAPONS)]


def get_weapon_count():
    """Get total number of weapons."""
    return len(WEAPONS)


def get_weapon_names():
    """Get list of all weapon names."""
    return [w.name for w in WEAPONS]
