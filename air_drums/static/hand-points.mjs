const HAND_WRIST = 0;
const INDEX_TIP = 8;

function distance(point, wrist, aspectRatio) {
  return Math.hypot((point.x - wrist.x) * aspectRatio, point.y - wrist.y);
}

function detection(result, index) {
  const landmarks = result?.landmarks?.[index];
  const category = result?.handedness?.[index]?.[0] ?? result?.handednesses?.[index]?.[0];
  const wrist = landmarks?.[HAND_WRIST];
  const tip = landmarks?.[INDEX_TIP];
  if (!wrist || !tip || !Number.isFinite(tip.x) || !Number.isFinite(tip.y)) return null;
  return {
    wrist,
    tip: {
      x: Math.max(0, Math.min(1, tip.x)),
      y: Math.max(0, Math.min(1, tip.y)),
      z: tip.z ?? 0,
      visibility: Math.max(0, Math.min(1, category?.score ?? 0.5)),
    },
  };
}

// Match each detected hand to the nearer Pose wrist so crossed or mirrored
// camera views do not depend on the hand model's handedness convention.
export function mapIndexTips(result, pose, aspectRatio = 4 / 3) {
  const tips = { leftIndex: null, rightIndex: null };
  const hands = (result?.landmarks ?? []).map((_, index) => detection(result, index)).filter(Boolean).slice(0, 2);
  const wrists = ['left', 'right'].map(hand => ({
    hand,
    point: pose?.[`${hand}Wrist`],
  })).filter(item => item.point && (item.point.visibility ?? 1) >= 0.35);

  if (hands.length === 1 && wrists.length) {
    const nearest = wrists.map(item => ({
      ...item,
      distance: distance(hands[0].wrist, item.point, aspectRatio),
    })).sort((a, b) => a.distance - b.distance)[0];
    tips[`${nearest.hand}Index`] = hands[0].tip;
    return tips;
  }

  if (hands.length === 2 && wrists.length === 1) {
    const closest = hands.map((hand, index) => ({
      index,
      distance: distance(hand.wrist, wrists[0].point, aspectRatio),
    })).sort((a, b) => a.distance - b.distance)[0];
    tips[`${wrists[0].hand}Index`] = hands[closest.index].tip;
    return tips;
  }

  if (hands.length === 2 && wrists.length === 2) {
    const direct = distance(hands[0].wrist, wrists[0].point, aspectRatio)
      + distance(hands[1].wrist, wrists[1].point, aspectRatio);
    const crossed = distance(hands[0].wrist, wrists[1].point, aspectRatio)
      + distance(hands[1].wrist, wrists[0].point, aspectRatio);
    const assignment = direct <= crossed ? [0, 1] : [1, 0];
    for (let handIndex = 0; handIndex < 2; handIndex += 1) {
      tips[`${wrists[assignment[handIndex]].hand}Index`] = hands[handIndex].tip;
    }
  }
  return tips;
}
