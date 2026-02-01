import { HomeCard } from '@/components/home-card';
import { MoreButton } from '@/components/more-button';
import { Routes } from '@/routes';
import { memo, useCallback } from 'react';
import { useNavigate } from 'react-router';
import { ISearchAppProps } from './hooks';
import { SearchDropdown } from './search-dropdown';

interface IProps {
  data: ISearchAppProps;
  showSearchRenameModal: (data: ISearchAppProps) => void;
}

// Memoized to prevent unnecessary re-renders when the parent list updates
export const SearchCard = memo(({ data, showSearchRenameModal }: IProps) => {
  const navigate = useNavigate();

  const handleClick = useCallback(() => {
    navigate(`${Routes.Search}/${data?.id}`);
  }, [navigate, data?.id]);

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
      onClick={handleClick}
    />
  );
});

SearchCard.displayName = 'SearchCard';
