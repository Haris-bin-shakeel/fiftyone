import { Selector } from "@fiftyone/components";
import * as fos from "@fiftyone/state";
import { useAtomValue } from "jotai";
import { useCallback, useMemo } from "react";
import { useRecoilState } from "recoil";
import styled from "styled-components";
import { isEditing } from "./Edit";
import { useApplyAnnotationSliceVisibility } from "./useApplyAnnotationSliceVisibility";
import type { AnnotationSliceInfo } from "./useGroupAnnotationSlices";
import { useGroupAnnotationSlices } from "./useGroupAnnotationSlices";

const Container = styled.div`
  padding: 0 1rem 0.5rem 1.5rem;
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const Label = styled.div`
  color: ${({ theme }) => theme.text.secondary};
  font-size: 1rem;
  white-space: nowrap;
`;

interface SliceOptionProps extends AnnotationSliceInfo {
  value: string;
}

const SliceOption = ({ value, mediaType, ...info }: SliceOptionProps) => {
  const isDisabled = info.isMissing || !info.isSupported;
  return (
    <span
      style={{
        opacity: isDisabled ? 0.5 : 1,
        cursor: isDisabled ? "not-allowed" : "pointer",
      }}
      title={
        !info.isSupported
          ? `${
              mediaType ? `"${mediaType}"` : "This"
            } media type does not support annotation`
          : info.isMissing
          ? `${value} slice does not exist"`
          : undefined
      }
    >
      {value}
      {!info.isSupported
        ? " (unsupported)"
        : info.isMissing
        ? " (missing)"
        : null}
    </span>
  );
};

const SliceSelector = ({
  onSliceSelected,
  slices,
}: GroupAnnotationProps & { slices: AnnotationSliceInfo[] | "loading" }) => {
  const isEditing_ = useAtomValue(isEditing);
  const [modalGroupSlice, setModalGroupSlice] = useRecoilState(
    fos.modalGroupSlice
  );
  const applyVisibilityForSlice = useApplyAnnotationSliceVisibility();
  const [preferredSlice, setPreferredSlice] =
    fos.usePreferredGroupAnnotationSlice();

  const allSlices = slices === "loading" ? [] : slices;

  const useSearch = useCallback(
    (search: string) => {
      const values = allSlices
        .filter(
          ({ name, isMissing, isSupported }) =>
            !isMissing &&
            isSupported &&
            name.toLowerCase().includes(search.toLowerCase())
        )
        .map((slice) => slice.name);
      return { values, total: values.length };
    },
    [allSlices]
  );

  const onSelect = useCallback(
    async (sliceName: string) => {
      const sliceInfo = allSlices.find((s) => s.name === sliceName);
      if (!sliceInfo?.isSupported || sliceInfo?.isMissing) {
        return modalGroupSlice;
      }

      applyVisibilityForSlice(sliceName);
      onSliceSelected?.();
      setModalGroupSlice(sliceName);
      setPreferredSlice(sliceName);
      return sliceName;
    },
    [
      allSlices,
      applyVisibilityForSlice,
      modalGroupSlice,
      onSliceSelected,
      setModalGroupSlice,
      setPreferredSlice,
    ]
  );

  const sliceInfoMap = useMemo(
    () => Object.fromEntries(allSlices.map((s) => [s.name, s])),
    [allSlices]
  );

  const SliceOptionComponent = useMemo(
    () =>
      ({ value }: { value: string }) => {
        const info = sliceInfoMap[value];
        return <SliceOption value={value} {...info} />;
      },
    [sliceInfoMap]
  );

  if (isEditing_ || (slices !== "loading" && allSlices.length === 0)) {
    return null;
  }

  return (
    <Container data-cy="annotation-slice-selector">
      <Selector
        inputStyle={{ height: 28, width: "100%" }}
        containerStyle={{ flex: 1 }}
        component={SliceOptionComponent}
        onSelect={onSelect}
        overflow={true}
        placeholder={slices === "loading" ? "Loading..." : "Select slice..."}
        useSearch={useSearch}
        resultsPlacement="bottom-start"
        value={slices === "loading" ? null : modalGroupSlice ?? preferredSlice}
        cy="annotation-slice"
      />
    </Container>
  );
};

interface GroupAnnotationProps {
  onSliceSelected?: () => void;
}

export default function GroupAnnotation({
  onSliceSelected,
}: GroupAnnotationProps) {
  const { resolved: slices } = useGroupAnnotationSlices();

  return <SliceSelector onSliceSelected={onSliceSelected} slices={slices} />;
}
