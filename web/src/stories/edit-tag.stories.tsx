import EditTag from '@/components/edit-tag';
import type { Meta, StoryObj } from '@storybook/react-webpack5';
import { useState } from 'react';
import { fn } from 'storybook/test';

const meta = {
  title: 'Components/EditTag',
  component: EditTag,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    value: { control: 'object' },
    disabled: { control: 'boolean' },
  },
  args: {
    onChange: fn(),
  },
} satisfies Meta<typeof EditTag>;

export default meta;
type Story = StoryObj<typeof meta>;

const DefaultStoryComponent = (args: any) => {
  const [tags, setTags] = useState(args.value || ['tag1', 'tag2']);

  return (
    <EditTag
      {...args}
      value={tags}
      onChange={(newTags) => {
        args.onChange?.(newTags);
        setTags(newTags);
      }}
    />
  );
};

export const Default: Story = {
  render: (args) => <DefaultStoryComponent {...args} />,
  args: {
    value: ['React', 'TypeScript', 'Tailwind'],
    disabled: false,
  },
};
