import type { DebateTurn } from "../types";

export const DEBATE_SCRIPT: DebateTurn[] = [
  { id: 0, speakerID: "A", text: "Pineapple on pizza is objectively CORRECT. It provides a sweet-savory contrast that elevates the entire dish. Your primitive palate simply cannot comprehend the complexity!", pitch: 1.4, rate: 1.15 },
  { id: 1, speakerID: "B", text: "That is perhaps the most CATASTROPHICALLY wrong statement ever uttered in human history. Pineapple destroys the structural integrity of the cheese and introduces moisture that makes the whole thing a soggy disaster.", pitch: 0.75, rate: 0.9 },
  { id: 2, speakerID: "A", text: "MOISTURE? You want to talk about MOISTURE? Every single topping releases moisture! Should we ban tomatoes too? Ban everything wet? This is pure pizza fascism and I will NOT stand for it!", pitch: 1.5, rate: 1.2 },
  { id: 3, speakerID: "B", text: "Pizza was perfected by the Italians. The Italians do not use fruit. There is a REASON for this. Centuries of culinary wisdom, my friend. Centuries. You are spitting in the face of Naples.", pitch: 0.7, rate: 0.85 },
  { id: 4, speakerID: "A", text: "Oh so NOW you are invoking NATIONALITY to win a food argument? Bold strategy! Tomatoes are from Mexico, mozzarella is from water buffalo, and YOUR logic is from the garbage!", pitch: 1.45, rate: 1.25 },
  { id: 5, speakerID: "B", text: "The tomato was ADOPTED and PERFECTED over centuries. Pineapple was introduced in nineteen-sixty-two by a Canadian man in a diner. A. Canadian. Man. This is not a debate, this is a crime scene investigation.", pitch: 0.72, rate: 0.88 },
  { id: 6, speakerID: "A", text: "And Beethoven was German and music was ALSO perfected by non-Italians so your entire argument just COLLAPSED under its own weight! Hawaiian pizza outsells your sad boring margherita every single weekend!", pitch: 1.55, rate: 1.3 },
  { id: 7, speakerID: "B", text: "Big Macs outsell every Michelin-starred restaurant on earth. Does that make them haute cuisine? Popularity is not validity. I rest my case. The defense rests. Court is adjourned.", pitch: 0.68, rate: 0.82 },
];