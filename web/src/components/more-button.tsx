import { useTranslate } from '@/hooks/common-hooks';
import { cn } from '@/lib/utils';
import { Ellipsis } from 'lucide-react';
import React from 'react';
import { Button, ButtonProps } from './ui/button';

export const MoreButton = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, size, ...props }, ref) => {
    const { t } = useTranslate('common');
    return (
      <Button
        ref={ref}
        variant="ghost"
        size={size || 'icon'}
        aria-label={props['aria-label'] || t('action')}
        className={cn(
          'opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto size-3.5 bg-transparent group-hover:bg-transparent',
          className,
        )}
        {...props}
      >
        <Ellipsis />
      </Button>
    );
  },
);
