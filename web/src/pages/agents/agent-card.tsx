import { HomeCard } from '@/components/home-card';
import { MoreButton } from '@/components/more-button';
import { SharedBadge } from '@/components/shared-badge';
import { Button } from '@/components/ui/button';
import { AgentCategory, AgentQuery } from '@/constants/agent';
import { IFlow } from '@/interfaces/database/agent';
import { Routes } from '@/routes';
import { Route } from 'lucide-react';
import { memo, useCallback } from 'react';
import { useNavigate } from 'react-router';
import { AgentDropdown } from './agent-dropdown';
import { useRenameAgent } from './use-rename-agent';

export type DatasetCardProps = {
  data: IFlow;
} & Pick<ReturnType<typeof useRenameAgent>, 'showAgentRenameModal'>;

// Memoized to prevent unnecessary re-renders when the parent list updates
export const AgentCard = memo(
  ({ data, showAgentRenameModal }: DatasetCardProps) => {
    const navigate = useNavigate();

    const handleClick = useCallback(() => {
      const id = data?.id;
      const category = data.canvas_category as AgentCategory;
      navigate(`${Routes.Agent}/${id}?${AgentQuery.Category}=${category}`);
    }, [navigate, data?.id, data.canvas_category]);

    return (
      <HomeCard
        data={{
          ...data,
          name: data.title,
          description: data.description || '',
        }}
        moreDropdown={
          <AgentDropdown
            showAgentRenameModal={showAgentRenameModal}
            agent={data}
          >
            <MoreButton></MoreButton>
          </AgentDropdown>
        }
        sharedBadge={<SharedBadge>{data.nickname}</SharedBadge>}
        onClick={handleClick}
        icon={
          data.canvas_category === AgentCategory.DataflowCanvas && (
            <Button variant={'ghost'} size={'sm'}>
              <Route />
            </Button>
          )
        }
      />
    );
  },
);

AgentCard.displayName = 'AgentCard';
