#!/usr/bin/env bash
set -euo pipefail

# Required inputs are explicit so a build cannot silently use another Hub
# revision or overwrite an earlier artifact.
: "${UMI_YAM_SOURCE_ROOT:?set to the pinned dual-lidar-umi-filtered-ad1db root}"
: "${UMI_YAM_ONSET_V2_ROOT:?set to the immutable onset-v2 artifact root}"
: "${UMI_YAM_ONSET_V3_ROOT:?set to a new onset-v3 output path}"

task_python="${UMI_YAM_PYTHON:-python}"
report_path="${UMI_YAM_ONSET_V3_REPORT:-${UMI_YAM_ONSET_V3_ROOT}.validation.json}"
final_parent="$(dirname -- "${UMI_YAM_ONSET_V3_ROOT}")"
final_name="$(basename -- "${UMI_YAM_ONSET_V3_ROOT}")"
mkdir -p "${final_parent}"

case "${report_path}" in
  "${UMI_YAM_ONSET_V3_ROOT}"/*)
    echo "Validation report must be outside the read-only artifact root" >&2
    exit 2
    ;;
esac

exec 9>"${final_parent}/.${final_name}.publish.lock"
flock 9
if [[ -e "${UMI_YAM_ONSET_V3_ROOT}" ]]; then
  echo "Refusing to overwrite ${UMI_YAM_ONSET_V3_ROOT}" >&2
  exit 2
fi

tree_digest() {
  local tree_root="$1"
  (
    cd "${tree_root}"
    find -L data meta videos -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | cut -d' ' -f1
  )
}

source_digest_before="$(tree_digest "${UMI_YAM_SOURCE_ROOT}")"
v2_digest_before="$(tree_digest "${UMI_YAM_ONSET_V2_ROOT}")"
stage_control="$(mktemp -d "${final_parent}/.${final_name}.build.XXXXXX")"
# The artifact staging root must be a sibling of the final root. Linux requires
# write permission on a moved directory when its parent changes (to update
# `..`), which conflicts with making the tree read-only before publication.
# A same-parent rename has no such requirement and remains atomic.
stage_root="${stage_control}.artifact"
prepublish_report="${stage_control}/prepublish-validation.json"
published=0

cleanup() {
  if [[ "${published}" -eq 0 && -d "${stage_root}" ]]; then
    chmod -R u+w "${stage_root}" 2>/dev/null || true
    rm -rf -- "${stage_root}"
  fi
  if [[ -d "${stage_control}" ]]; then
    rm -rf -- "${stage_control}"
  fi
}
trap cleanup EXIT

if [[ -e "${stage_root}" ]]; then
  echo "Reserved staging artifact path unexpectedly exists" >&2
  exit 2
fi
if [[ "$(stat -c %d "${stage_control}")" != "$(stat -c %d "${final_parent}")" ]]; then
  echo "Staging and final roots must be on the same filesystem" >&2
  exit 2
fi

"${task_python}" -m lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v3 \
  --source-root="${UMI_YAM_SOURCE_ROOT}" \
  --output-root="${stage_root}"

"${task_python}" examples/umi_yam/validate_currentrel_onset_v3_artifact.py \
  --source-root="${UMI_YAM_SOURCE_ROOT}" \
  --v2-root="${UMI_YAM_ONSET_V2_ROOT}" \
  --v3-root="${stage_root}" \
  --report="${prepublish_report}"

source_digest_after_build="$(tree_digest "${UMI_YAM_SOURCE_ROOT}")"
v2_digest_after_build="$(tree_digest "${UMI_YAM_ONSET_V2_ROOT}")"
if [[ "${source_digest_before}" != "${source_digest_after_build}" ]]; then
  echo "Pinned source changed during onset-v3 build" >&2
  exit 2
fi
if [[ "${v2_digest_before}" != "${v2_digest_after_build}" ]]; then
  echo "Immutable onset-v2 parent changed during onset-v3 build" >&2
  exit 2
fi

chmod -R a-w "${stage_root}"
if find "${stage_root}" -perm /222 -print -quit | grep -q .; then
  echo "Staged artifact still contains writable paths" >&2
  exit 2
fi
if [[ -e "${UMI_YAM_ONSET_V3_ROOT}" ]]; then
  echo "Final artifact appeared while staging; refusing to publish" >&2
  exit 2
fi
mv -T -- "${stage_root}" "${UMI_YAM_ONSET_V3_ROOT}"
published=1

"${task_python}" examples/umi_yam/validate_currentrel_onset_v3_artifact.py \
  --source-root="${UMI_YAM_SOURCE_ROOT}" \
  --v2-root="${UMI_YAM_ONSET_V2_ROOT}" \
  --v3-root="${UMI_YAM_ONSET_V3_ROOT}" \
  --report="${report_path}"

source_digest_after_publish="$(tree_digest "${UMI_YAM_SOURCE_ROOT}")"
v2_digest_after_publish="$(tree_digest "${UMI_YAM_ONSET_V2_ROOT}")"
if [[ "${source_digest_before}" != "${source_digest_after_publish}" ]]; then
  echo "Pinned source changed during onset-v3 publish" >&2
  exit 2
fi
if [[ "${v2_digest_before}" != "${v2_digest_after_publish}" ]]; then
  echo "Immutable onset-v2 parent changed during onset-v3 publish" >&2
  exit 2
fi

rm -rf -- "${stage_control}"
trap - EXIT

echo "Validated immutable onset-v3 artifact: ${UMI_YAM_ONSET_V3_ROOT}"
echo "Validation report: ${report_path}"
