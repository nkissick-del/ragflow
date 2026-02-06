import { HomeCard } from '@/components/home-card';
import { MoreButton } from '@/components/more-button';
import { useNavigatePage } from '@/hooks/logic-hooks/navigate-hooks';
import { memo } from 'react';
import { IMemory } from './interface';
import { MemoryDropdown } from './memory-dropdown';

interface IProps {
  data: IMemory;
  showMemoryRenameModal: (data: IMemory) => void;
}

// Memoized to prevent unnecessary re-renders when the parent list updates
export const MemoryCard = memo(({ data, showMemoryRenameModal }: IProps) => {
  const { navigateToMemory } = useNavigatePage();

  return (
    <HomeCard
      data={{
        name: data?.name,
        avatar: data?.avatar,
        description: data?.description,
        update_time: data?.create_time,
      }}
      moreDropdown={
        <MemoryDropdown
          memory={data}
          showMemoryRenameModal={showMemoryRenameModal}
        >
          <MoreButton></MoreButton>
        </MemoryDropdown>
      }
      onClick={navigateToMemory(data?.id)}
    />
  );
});

MemoryCard.displayName = 'MemoryCard';
