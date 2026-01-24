import { HomeCard } from '@/components/home-card';
import { MoreButton } from '@/components/more-button';
import { useNavigatePage } from '@/hooks/logic-hooks/navigate-hooks';
import { memo } from 'react';
import { ISearchAppProps } from './hooks';
import { SearchDropdown } from './search-dropdown';

interface IProps {
  data: ISearchAppProps;
  showSearchRenameModal: (data: ISearchAppProps) => void;
}
// Optimized: Wrapped in React.memo to prevent unnecessary re-renders when parent state changes.
export const SearchCard = memo(({ data, showSearchRenameModal }: IProps) => {
  const { navigateToSearch } = useNavigatePage();

  return (
    <HomeCard
      data={data}
      moreDropdown={
        <SearchDropdown
          dataset={data}
          showSearchRenameModal={showSearchRenameModal}
        >
          <MoreButton></MoreButton>
        </SearchDropdown>
      }
      onClick={navigateToSearch(data?.id)}
    />
  );
});

SearchCard.displayName = 'SearchCard';
