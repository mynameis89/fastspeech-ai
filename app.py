import time
from inference.engine import generate_voice

if __name__ == "__main__":
    while True:
        print("🤖 Starting Think Dash voice generation...")

        # Yeh text aap Think Dash video ke liye customize kar sakte ho
        text = "Aaj ka vishay hai: Vigyan aur Soch. Like, Share, aur Subscribe karein Think Dash ko!"

        # Voice generation call
        output_path = generate_voice(text)

        print(f"✅ Voice generated successfully: {output_path}")

        # Sleep for 24 hours (86400 seconds)
        time.sleep(86400)